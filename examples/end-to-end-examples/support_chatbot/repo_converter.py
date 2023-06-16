# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

from git.repo import Repo


class RepoConverter:
    """Converts .md, .py, and .YAML files in git repositories to text files that
    land in /scripts/train/support_chatbot/data/{REPOSITORY_NAME}

    Args:
        output_dir (str): The path of the directory where the converted repository will be saved
        repo_url (str): The url for the git repository

    Attributes:
        output_dir (str): The path of the directory where the converted repository will be saved
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
            converter = RepoConverter(repo_url)
            converter.convert_repo()
    """

    def __init__(self, output_dir: (str), 
                 current_dir: (str),
                 repo_url: (str)) -> None:
        
        self.output_dir = output_dir
        self.repo_url = repo_url
        self.repo_name = repo_url.split('/')[-1]
        self.clone_dir = os.path.join(current_dir, self.repo_name)

        if os.path.exists(self.clone_dir):
            raise ValueError(f"{self.clone_dir} already exists. Please choose a path that doesn't contain the repository name.")

    def prepare_output_file(self, file_path: (str)) -> str:
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

        relative_path = os.path.relpath(file_path, self.clone_dir).replace('.yaml', '').replace('.py', '').replace('.md', '')
        relative_path = relative_path.replace('/', '_')
        output_file = os.path.join(self.output_dir, self.repo_name, relative_path + '.txt')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        return output_file
    
    def yaml_to_txt(self, file_path: (str)) -> None:
        """Given the file_path of a .YAML file in cloned repository, converts it
        to a .txt file and saves it in the same directory structure in
        /scripts/train/support_chatbot/data/{self.repo_name}

        Args:
            file_path (str): the file_path of a .YAML file in cloned repository
        """
        with open(file_path, 'r') as file:
                yaml_content = file.read()  # read file as regular text file
        output_file = self.prepare_output_file(file_path)
        with open(output_file, 'w') as out_file:
            out_file.write(yaml_content)  # write the content to the output file

    def py_to_txt(self, file_path: (str)) -> None:
        """Given the file_path of a .py file in cloned repository, converts it
        to a .txt file and saves it in the same directory structure in
        /scripts/train/support_chatbot/data/{self.repo_name}

        Args:
            file_path (str): the file_path of a .py file in cloned repository
        """
        with open(file_path, 'r') as f:
            code_content = f.read()
        output_file = self.prepare_output_file(file_path)
        with open(output_file, 'w') as out_file:
            out_file.write(code_content)

    def md_to_txt(self, file_path: (str)) -> None:
        """Given the file_path of a .py file in cloned repository, converts it
        to a .md file and saves it in the same directory structure in
        /scripts/train/support_chatbot/data/{self.repo_name}

        Args:
            file_path (str): the file_path of a .md file in cloned repository
        """
        with open(file_path, 'r') as file:
            md_content = file.read()
        output_file = self.prepare_output_file(file_path)
        with open(output_file, 'w') as out_file:
            out_file.write(md_content)

    def convert_to_txt(self, file_path: (str)) -> None:
        """Given a file path in cloned repository, runs the appropriate
        conversion function based on the file extension.

        Args:
            file_path (str): the file_path in cloned repository
        """
        _, ext = os.path.splitext(file_path)
        if ext == '.yaml':
            self.yaml_to_txt(file_path)
        elif ext == '.py':
            self.py_to_txt(file_path)
        elif ext == '.md':
            self.md_to_txt(file_path)
        else:
            print(f'Unsupported file type: {ext}')

    def convert_repo(self) -> str:
        """Given a git repository url clone the repository, then convert all
        repository .yaml, .py, and .md files to .txt files and save them in
        /scripts/train/support_chatbot/data/{self.repo_name}

        Returns:
            The path of the converted repository (/scripts/train/support_chatbot/data/{self.repo_name})
        """
        # Cloning the repo
        Repo.clone_from(self.repo_url, self.clone_dir)

        # Converting each file
        for root, _, files in os.walk(self.clone_dir):
            for file in files:
                if file.endswith(('.yaml', '.py', '.md')):
                    full_file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(full_file_path)
                    if ext == '.yaml':
                        self.yaml_to_txt(full_file_path)
                    elif ext == '.py':
                        self.py_to_txt(full_file_path)
                    elif ext == '.md':
                        self.md_to_txt(full_file_path)
                    else:
                        print(f'Unsupported file type: {ext}')

        shutil.rmtree(self.clone_dir)
        return os.path.join(self.output_dir, self.repo_name)
