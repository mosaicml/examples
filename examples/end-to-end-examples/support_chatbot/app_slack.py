import os
import time
import random
import string
from io import BytesIO
import hashlib
import hmac

from argparse import ArgumentParser, Namespace
from langchain.embeddings import MosaicMLInstructorEmbeddings
from langchain.llms import MosaicML
from chatbot import ChatBot

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from flask import Flask, request, jsonify
from pyngrok import ngrok

from oci_converser import OCIObjectStorageManager

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

BOT_USER_ID = "U05MKGP6J84"
processed_events = set()

REDIRECT_URI = 'http://localhost:3000/login/callback'

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(description='Run a chatbot!')
    parser.add_argument('--endpoint_url', type=str, default='https://models.hosted-on.mosaicml.hosting/mpt-30b-chat/v1/predict', required=False, help='The endpoint of our MosaicML LLM Model')
    parser.add_argument('--max_length', type=int, default=1200, required=False, help='The maximum size of context from LangChain')
    parser.add_argument('--chunk_size', type=int, default=1200, required=False, help='The chunk size when splitting documents')
    parser.add_argument('--chunk_overlap', type=int, default=800, required=False, help='The overlap between chunks when splitting documents')
    parser.add_argument('--retrieval_k', type=int, default=5, required=False, help='The number of chunks to retrieve as context from vector store')
    parser.add_argument('--model_k', type=int, default=10, required=False, help='The number of outputs model should output')
    parser.add_argument('--repository_urls', type=str, default='https://github.com/mosaicml/composer,https://github.com/mosaicml/streaming,https://github.com/mosaicml/examples,https://github.com/mosaicml/diffusion,https://github.com/mosaicml/llm-foundry', required=False, help='The GitHub repository URLs to download')
    parser.add_argument('--data_collecting', type=bool, default=False, help='Where successful threads will be stored')
    parser.add_argument('--slack_token', type=str, help='Slack Token')
    parser.add_argument('--slack_signing_secret', type=str, help='Slack Signing Secret')
    parser.add_argument('--oci_data_storage', type=str, default='oci://mosaicml-internal-checkpoints/support-bot-demo/slack-data', help='Where successful threads will be stored')
    parser.add_argument('--complex_chat', type=bool, default=False, help='Where successful threads will be stored')

    parsed = parser.parse_args()
    if parsed.repository_urls is not None:
        parsed.repository_urls = ''.join(str(parsed.repository_urls).split()).split(',')
    return parsed

@app.route('/slack/events', methods=['POST'])
def slack_events():
    data = request.json
    
    # Immediately respond to Slack's challenge
    if "challenge" in data:
        return jsonify({'challenge': data['challenge']})
    
    # After challenge check, verify the Slack request
    if not verify_slack_request(request):
        return jsonify({'message': 'Unauthorized'}), 401
    
    # Deduplication using event timestamp
    event_ts = data['event'].get('event_ts', None)
    if event_ts in processed_events:
        return jsonify({'status': 'already_processed'})
    processed_events.add(event_ts)

    if 'text' in data['event'] and data['event']['type'] == 'message' and f"<@{BOT_USER_ID}>" in data['event']['text']:
        channel_id = data['event']['channel']
        thread_ts = data['event'].get('thread_ts', data['event']['ts']) # Default to current message TS if not a thread

        # Fetch entire thread using Slack's API
        thread_messages = client.conversations_replies(channel=channel_id, ts=thread_ts)['messages']

        # Process thread messages
        conversation_msgs = []
        question_msg = None
        previous_msg = None  # Store the previous message
        for msg in thread_messages:
            user_id = msg['user']
            user_info = client.users_info(user=user_id)
            user_name = user_info['user']['name']
            formatted_msg = f"{user_name}: {msg['text']}."

            # Check if the message is just the bot ping
            if msg['text'].strip() == f"<@{BOT_USER_ID}>":
                if previous_msg:
                    question_msg = previous_msg
                    previous_msg = None  # Reset previous_msg so it doesn't get added to conversation_msgs
                continue

            # Separate the message with the ping
            if f"<@{BOT_USER_ID}>" in msg['text']:
                question_msg = formatted_msg
            else:
                conversation_msgs.append(formatted_msg)
                previous_msg = formatted_msg

        # If the question_msg is still None after the loop, it means the bot was pinged without any question. 
        # Set the last message (if any) in the thread as the question in such cases.
        if question_msg is None and previous_msg:
            question_msg = previous_msg
            conversation_msgs.remove(previous_msg)  # Remove it from the context as it's now the question
        
        # Construct the message for the model
        conversation = " ".join(conversation_msgs)
        if len(conversation) > 2000:
            conversation = conversation[-2000:]
        user_msg = f"Here is the conversation so far: {conversation} Here is the question: {question_msg}"
        
        print(user_msg)

        # Respond quickly to Slack
        response = jsonify({'status': 'acknowledged'})
        if chat_version:
            chat_response = chatbot.sub_query_chat(user_msg)
        else:
            chat_response = chatbot.chat(user_msg)
        
        # Post response in the same thread
        post_args = {'channel': channel_id, 'text': chat_response, 'thread_ts': thread_ts}
        
        try:
            client.chat_postMessage(**post_args)
        except SlackApiError as e:
            print(f"Slack API Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")

        return response
    # Handling reactions added to messages
    elif data['event']['type'] == 'reaction_added':
        print(f"Detected reaction: {data['event']['reaction']}")
        if data['event']['reaction'] == 'white_check_mark':  # Checkmark reaction
            channel_id = data['event']['item']['channel']
            message_ts = data['event']['item']['ts']
            
            # Fetch the entire thread related to the reacted message
            root_msg = get_root_message(ts=message_ts, channel=channel_id, need_thread_ts=True)
            thread_messages = client.conversations_replies(channel=channel_id, ts=root_msg["thread_ts"])['messages']
            
            if contains_checkmark(thread_messages):
                save_thread_to_oci(thread_messages, root_msg)

    elif data['event']['type'] == 'reaction_removed':
        print(f"Detected reaction removal: {data['event']['reaction']}")
        if data['event']['reaction'] == 'white_check_mark':  # Checkmark reaction removed
            channel_id = data['event']['item']['channel']
            message_ts = data['event']['item']['ts']
                
            # Fetch the entire thread related to the reacted message
            root_msg = get_root_message(ts=message_ts, channel=channel_id, need_thread_ts=True)
            thread_messages = client.conversations_replies(channel=channel_id, ts=root_msg["thread_ts"])['messages']
            remove_thread_from_oci(root_msg)
            if contains_checkmark(thread_messages):
                # If another checkmark is still present, re-save the new updated thread again
                save_thread_to_oci(thread_messages, root_msg)

    return jsonify({'status': 'ok'})

def contains_checkmark(thread_messages):
    for msg in thread_messages:
        if 'reactions' in msg:
            for reaction in msg['reactions']:
                if reaction['name'] == 'white_check_mark':
                    return True
    return False

def send_slack_message(client, channel, message):
    """Send a message to a Slack channel."""
    try:
        client.chat_postMessage(channel=channel, text=message)
    except SlackApiError as e:
        print(f"Slack API Error: {e}")

def generate_random_name(length=5):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

def save_thread_to_oci(thread_messages, root_msg):
    buffer = BytesIO()
    
    # Dictionary to store the mapping of original user names to randomized user names
    user_name_mapping = {}
    
    for msg in thread_messages:
        user_id = msg['user']
        user_name = client.users_info(user=user_id)['user']['name']

        # Check if the user name already has a randomized counterpart
        # If not, generate one and store in the dictionary
        if user_name not in user_name_mapping:
            user_name_mapping[user_name] = generate_random_name()
        
        # Replace the original user name with its randomized version
        randomized_user_name = user_name_mapping[user_name]
        
        if 'reactions' in msg:
            for reaction in msg['reactions']:
                if reaction['name'] == 'white_check_mark':
                    buffer.write(f"Accepted Answer by {randomized_user_name}: {msg['text']}\n".encode('utf-8'))
                else:
                    buffer.write(f"(Context) {randomized_user_name}: {msg['text']}\n".encode('utf-8'))
        else:
            buffer.write(f"(Context) {randomized_user_name}: {msg['text']}\n".encode('utf-8'))
    
    # Set the pointer to the beginning of the BytesIO object
    buffer.seek(0)

    # Now, upload to OCI
    timestamp = root_msg['ts']
    formatted_time = time.strftime('message_from_slack_%Y%m%d%H%M%S', time.gmtime(float(timestamp)))
    object_name = f'text{formatted_time}.txt'

    oci_manager.upload_file_obj(buffer, object_name)

def remove_thread_from_oci(root_msg):
    timestamp = root_msg['ts']
    formatted_time = time.strftime('%Y%m%d%H%M%S', time.gmtime(float(timestamp)))
    object_name = f'text{formatted_time}.txt'
    oci_manager.delete_file_obj(object_name)

def get_root_message(ts, channel, need_thread_ts=False):
    resp = client.conversations_replies(ts=ts, channel=channel)
    first_msg = resp["messages"][0]
    if not need_thread_ts:
        return first_msg
    return get_root_message(first_msg["thread_ts"], channel, need_thread_ts=False)

def verify_slack_request(request):
    """
    Verifies that the POST request comes from Slack.
    """
    signature = request.headers.get("X-Slack-Signature")
    timestamp = request.headers.get("X-Slack-Request-Timestamp")

    # Avoid replay attacks
    if abs(time.time() - int(timestamp)) > 60 * 5:  # 5 minutes
        return False

    # Form the basestring as defined by Slack
    basestring = f"v0:{timestamp}:{request.get_data().decode('utf-8')}"

    # Hash the basestring
    my_signature = 'v0=' + hmac.new(
        bytes(signing_secret, 'utf-8'),
        msg=bytes(basestring, 'utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(my_signature, signature)

def main(endpoint_url: str, 
         max_length: int, 
         chunk_size: int, 
         chunk_overlap: int, 
         retrieval_k: int, 
         model_k: int, 
         repository_urls: list[str], 
         data_collecting: bool,
         slack_token: str, 
         slack_signing_secret: str,
         oci_data_storage: str,
         complex_chat: bool):
    
    if slack_token is None:
        try:
            slack_token = os.environ["COMPOSER_BOT_SLACK_TOKEN"]
        except KeyError:
            ValueError('No slack token provided. Please provide a slack token or set the SLACK_BOT_TOKEN environment variable')

    if slack_signing_secret is None:
        try:
            slack_signing_secret = os.environ["SLACK_SIGNING_SECRET"]
        except KeyError:
            ValueError('No slack signing secret provided. Please provide a slack signing secret or set the SLACK_BOT_TOKEN environment variable')
    
    global chatbot, client, oci_manager, read_slack, signing_secret, chat_version
    oci_manager = OCIObjectStorageManager(oci_uri=oci_data_storage)
    read_slack = data_collecting
    signing_secret = slack_signing_secret
    chat_version = complex_chat

    retrieval_dir = os.path.join(ROOT_DIR, 'retrieval_data_slack')

    embeddings = MosaicMLInstructorEmbeddings()
    llm = MosaicML(
        inject_instruction_format=True,
        endpoint_url=endpoint_url,
        model_kwargs={'output_len': max_length, 'top_k': model_k, 'top_p': 0.95, 'temperature': 0.1}
    )
    
    chatbot = ChatBot(data_path=retrieval_dir, 
                      embedding=embeddings,
                      model=llm, 
                      k=retrieval_k, 
                      chunk_size=chunk_size, 
                      chunk_overlap=chunk_overlap,
                      slack_path=oci_data_storage)

    if not os.path.isfile(os.path.join(retrieval_dir, 'vectors.pickle')):
        if repository_urls is None:
            raise ValueError('No repository URLs provided. Please provide a comma separated list of URLs to download')  
        chatbot.create_vector_store(repository_urls=repository_urls)

    client = WebClient(token=slack_token)
    existing_tunnels = ngrok.get_tunnels()
    for tunnel in existing_tunnels:
        if "http://127.0.0.1:3000" in tunnel.public_url:
            public_url = tunnel.public_url
            break
    else:
        public_url = ngrok.connect(3000)
    print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 3000))

    for rule in app.url_map.iter_rules():
        print(rule)

    app.run(port=3000, debug=False)

if __name__ == "__main__":
    args = parse_args()
    main(endpoint_url=args.endpoint_url, 
         max_length=args.max_length, 
         chunk_size=args.chunk_size, 
         chunk_overlap=args.chunk_overlap, 
         retrieval_k=args.retrieval_k, 
         model_k=args.model_k, 
         repository_urls=args.repository_urls, 
         data_collecting=args.data_collecting,
         slack_token=args.slack_token, 
         slack_signing_secret=args.slack_signing_secret, 
         oci_data_storage=args.oci_data_storage,
         complex_chat=args.complex_chat)