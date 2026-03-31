# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys

from git.repo import Repo


class RepoDownloader:
    """Downloads .md, .py, and .YAML files in git repositories to text files
    that land in /scripts/train/support_chatbot/retrieval_data/{REPOSITORY_NAME}

    Args:
        output_dir (str): The path of the directory where the downloaded repository will be saved
        repo_url (str): The url for the git repository

    Attributes:
        output_dir (str): The path of the directory where the downloaded repository will be saved
        repo_url (str): The url for the git repository
        repo_name (str): The name of the git repository
        clone_dir (str): The path of the directory where the git repository will be cloned

    Raises:
        ValueError: If the clone_dir (directory of os.path.join(current_dir, self.repo_name)) already exists

    Warning:
        Make sure to use the actual github link (example: https://github.com/KuuCi/test_repository)
        instead of the clone link which will end in '.git'

    Example:
    .. testcode::

        import sys

        for repo_url in sys.argv[1:]:
            downloader = RepoDownloader(repo_url)
            downloader.download_repo()
    """

    def __init__(self, output_dir: str, current_dir: str,
                 repo_url: str) -> None:

        self.output_dir = output_dir
        self.repo_url = repo_url
        self.repo_name = repo_url.split('/')[-1]
        self.clone_dir = os.path.join(current_dir, self.repo_name)

        if os.path.exists(self.clone_dir):
            raise ValueError(
                f"{self.clone_dir} already exists. Please choose a path that doesn't contain the repository name."
            )

    def get_github_file_url(self, file_path: str) -> str:
        """Generate GitHub URL for a specific file in the repository."""
        relative_path = os.path.relpath(file_path, self.clone_dir)
        # Ensure that the base GitHub URL is always included
        github_file_url = f"https://github.com/{self.repo_url.split('/')[-2]}/{self.repo_name}/blob/main/{relative_path}"
        return github_file_url

    def prepare_output_file(self, file_path: str) -> str:
        """Given the .py, .md, or .YAML file_path of the cloned git repository
        file, returns the path of the new txt processed output file and creates
        the new path's intermediate directory if it doesn't exist.

        Args:
            file_path (str): the path of a .py, .md, or .yaml file in cloned repository

        Raises:
            ValueError: If the file_path is not a .py, .md, or .yaml file

        Returns:
            str: the path of the .txt version of that file
        """
        _, ext = os.path.splitext(file_path)
        if ext not in ['.yaml', '.py', '.md']:
            raise ValueError(f'Unsupported file type: {ext}')

        github_url = self.get_github_file_url(file_path)

        # Convert the GitHub URL into the desired filename format
        filename = github_url.replace('/', '{slash}').replace('.', '{dot}')

        output_file = os.path.join(self.output_dir, self.repo_name,
                                   filename + '.txt')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        return output_file

    def file_to_txt(self, file_path: (str)) -> None:
        """Given the file_path of a file in cloned repository, downloads it to
        a.

        .txt file and saves it in the same directory structure in.

        /scripts/train/support_chatbot/retrieval_data/{self.repo_name}

        Args:
            file_path (str): the file_path of a .py file in cloned repository
        """
        with open(file_path, 'r') as f:
            code_content = f.read()
        output_file = self.prepare_output_file(file_path)
        with open(output_file, 'w') as out_file:
            out_file.write(code_content)

    def download_repo(self) -> str:
        """Given a git repository url clone the repository, then download all
        repository .yaml, .py, and .md files as .txt files and save them in.

        /scripts/train/support_chatbot/retrieval_data/{self.repo_name}

        Returns:
            The path of the downloaded repository (/scripts/train/support_chatbot/retrieval_data/{self.repo_name})
        """
        # Cloning the repo
        Repo.clone_from(self.repo_url, self.clone_dir)

        # Downloading each file
        for root, _, files in os.walk(self.clone_dir):
            for file in files:
                if file.endswith(('.yaml', '.py', '.md')):
                    full_file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(full_file_path)
                    if ext == '.yaml' or ext == '.py' or ext == '.md':
                        self.file_to_txt(full_file_path)
                    else:
                        print(f'Unsupported file type: {ext}')

        shutil.rmtree(self.clone_dir)
        return os.path.join(self.output_dir, self.repo_name)


def main() -> None:
    output_dir = 'retrieval_data'
    if len(sys.argv) < 2:
        raise ValueError(
            'At least one repository URL must be provided as an argument.')

    for repo_url in sys.argv[1:]:
        downloader = RepoDownloader(output_dir, '', repo_url)
        if os.path.exists(downloader.clone_dir):
            continue
        downloader.download_repo()


if __name__ == '__main__':
    main()
