
from attr import has
import torch
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F
import pickle

RESOURCES = {
    "gpt2": "src/speedups/artifacts/wordnet_lin_top_100_whole_word_only_gpt2_tokenizer.pkl",
    "bert-base-uncased": "llm/speedups/artifacts/wordnet_lin_top_100_bert_tokenizer.pkl"
}
class PartialCreditWarmupLoss:
    vocab_size: int
    top_k: int
    max_warmup_steps: int
    step_counter: int
    partial_credit_scaling_factor: float
    partial_credit_assignments: Dict[int, Tuple[int, float]]


    @classmethod
    def from_config(cls, cfg) -> "PartialCreditWarmupLoss":
        assert hasattr(cfg, "top_k")
        top_k = cfg.top_k
        assert hasattr(cfg, "max_warmup_steps")
        max_warmup_steps = cfg.max_warmup_steps
        assert hasattr(cfg, "partial_credit_scaling_factor")
        partial_credit_scaling_factor = cfg.partial_credit_scaling_factor


        assert hasattr(cfg, "tokenizer_name")

        if cfg.tokenizer_name in RESOURCES:
            partial_credit_dict_path = RESOURCES[cfg.tokenizer_name]
        else:
            raise Exception(f"Unrecognized tokenizer: {cfg.tokenizer_name} not found in available resources {RESOURCES}")
        pwc = PartialCreditWarmupLoss(
            top_k=top_k,
            max_warmup_steps=max_warmup_steps,
            partial_credit_scaling_factor=partial_credit_scaling_factor,
            partial_credit_dict_path=partial_credit_dict_path
        )

        return pwc

    def __init__(self,
        top_k: int,
        max_warmup_steps: int,
        partial_credit_scaling_factor: float,
        vocab_size: Optional[int] = None,
        partial_credit_dict_path: Optional[str] = None
    ):
        super(PartialCreditWarmupLoss, self).__init__()
        self.top_k = top_k
        self.step_counter = 0
        self.max_warmup_steps = max_warmup_steps
        self.partial_credit_scaling_factor = partial_credit_scaling_factor


        if partial_credit_dict_path is not None:
            with open(partial_credit_dict_path, "rb") as f:
                self.partial_credit_assignments = pickle.load(f)
            self.vocab_size  = len(self.partial_credit_assignments) 
        else:
            if vocab_size:
                self.vocab_size = vocab_size
            else:
                raise Exception("If a partial credit dict path is not provided, you must provide the vocab size manually")
            self.partial_credit_assignments = self.construct_mock_assignments()


        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")


    
    def construct_mock_assignments(self) -> Dict[int, Tuple[int, float]]:
        return {i: [(i, 1.0)] for i in range(0, self.vocab_size)}

    def scale_target_probabilities(self, baseline_target, partial_credit_target):
        scaling_multiplier = (1 - self.step_counter / self.max_warmup_steps) * self.partial_credit_scaling_factor
        return (1-scaling_multiplier) * baseline_target[:,1] + scaling_multiplier  * F.softmax(partial_credit_target[:,1], dim=0)

    def proba_distribution(self, x):
        baseline_target = torch.zeros(self.top_k, 2)
        baseline_target[:,1] = 0
        baseline_target[0] = torch.FloatTensor([x.item(), 1])
        res = torch.zeros(self.top_k, 2)
        if x.item() not in self.partial_credit_assignments:
            return baseline_target
        res = torch.zeros(self.top_k, 2)
        res[:,1] = float("-inf")


        # this way we always ensure that the real token is somewhere in the partial credit matrix
        res[0][0] = x.item()
        res[0][1] = 1.0

        idx = 1
        it = iter(self.partial_credit_assignments[x.item()])
        while idx < self.top_k:
            try:
                token, prob = next(it)
                if token == x.item():
                    continue
                else:
                    res[idx][0] = token
                    res[idx][1] = prob
                    idx += 1
            except:
                break


        
        res[:,1] = self.scale_target_probabilities(baseline_target, res)
        return res

    def make_partial_credit_tensor(self, tensor: torch.Tensor, device) -> torch.Tensor:
        bsz, seqlen = tensor.shape[0], tensor.shape[1]
        res = torch.stack([
            self.proba_distribution(x_i) for i, x_i in enumerate(torch.unbind(tensor.ravel(), dim=0), 0)
        ], axis=0).reshape(bsz, seqlen, self.top_k, 2).to(device)
        labels, probabilities = res.tensor_split(2, dim=3)
        return labels.reshape(bsz * seqlen, self.top_k).long(), probabilities.reshape(bsz * seqlen, self.top_k)


    def get_partial_credit_targets(self, targets: torch.LongTensor, device) -> torch.FloatTensor:
        """_summary_

        Args:
            targets (torch.LongTensor): (Bsz x Seqlen) Categorial targets representing the token
            expected at each sequence index in the batch

        Returns:
            torch.FloatTensor: (Bsz x Seqlen x vocab_size) A probability distribution representing the
            probability distribution induced by partial credit assignment for each sequence index in the batch
        """
        return self.make_partial_credit_tensor(targets, device)


    def get_targets(self, batch):
            targets = torch.roll(batch["labels"], shifts=-1)
            targets[:, -1] = -100
            return targets

    def _get_partial_credit_loss(self, outputs: torch.FloatTensor, targets: torch.LongTensor) -> torch.Tensor:
        """
        According to https://github.com/pytorch/pytorch/issues/11959, cross entropy w/ probabilistic labels
        can be implemented as softmax + KL divergence.

        Since we are taking the top_k and not the full vocabulary, its more efficient to calculate softmax,
        select the k indices we are interested in, then pass it the KLDivLoss.

        The alternative would be constructing a very sparse target vector with shape |batch| * |seq| * |vocab_size|
        where the last dimension only has k non-zero values. With this implementation, our target is of size
        |batch| * |seq| * k
        """
        
        device = outputs.device
        if self.step_counter >= self.max_warmup_steps:
            # default to regular cross entropy
            return F.cross_entropy(outputs.view(-1, outputs.size(-1)),
                               targets.view(-1),
                               ignore_index=-100)

        partial_credit_labels, partial_credit_probabilities = self.get_partial_credit_targets(targets, device)
        

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        outputs = F.log_softmax(outputs[targets != -100, :], dim=1)
        partial_credit_labels = partial_credit_labels[targets != -100, :].to(device)
        partial_credit_probabilities = partial_credit_probabilities[targets != -100, :].to(device)
        
        gathered_outputs = torch.gather(outputs, 1, partial_credit_labels)

  
        return self.kl_loss(gathered_outputs, partial_credit_probabilities)

    def loss(self, outputs: torch.FloatTensor, batch: Dict[str, torch.LongTensor]) -> torch.Tensor:
        targets = self.get_targets(batch)
        loss = self._get_partial_credit_loss(outputs, targets)
        self.finish_step()
        return loss


    def finish_step(self):
        self.step_counter += 1

        

