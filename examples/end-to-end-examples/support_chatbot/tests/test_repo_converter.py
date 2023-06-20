import os
import pytest
import sys
import shutil
from git.repo import Repo

DIR_PATH = 'examples/end-to-end-examples/support_chatbot'
sys.path.append(DIR_PATH)
from repo_converter import RepoConverter

@pytest.fixture
def converter():
    output_dir = 'examples/end-to-end-examples/support_chatbot/tests/test_data'
    current_dir = 'examples/end-to-end-examples/support_chatbot/tests'
    repo_url = 'https://github.com/KuuCi/test_repository'
    return RepoConverter(output_dir, current_dir, repo_url)

@pytest.mark.parametrize('file_path', ['examples/end-to-end-examples/support_chatbot/tests/test_repository/md_test.md', 
                                       'examples/end-to-end-examples/support_chatbot/tests/test_repository/folder_test/python_test.py', 
                                       'examples/end-to-end-examples/support_chatbot/tests/test_repository/folder_test/yaml_test.yaml',
                                       'examples/end-to-end-examples/support_chatbot/tests/test_repository/folder_test/unsupported_test.txt'
                                       ])
def test_converter_prepare_output_file(file_path: str, converter: RepoConverter):
    _, ext = os.path.splitext(file_path)
    if ext not in ['.yaml', '.py', '.md']:
        with pytest.raises(
                    ValueError,
                    match='Unsupported file type: .txt'):
            converter.prepare_output_file(file_path)
        return
    elif ext == '.md':
        file_name = 'md_test.txt'
    elif ext == '.py':
        file_name = 'folder_test_python_test.txt'
    else:
        file_name = 'folder_test_yaml_test.txt'        

    expected_output = os.path.join(converter.output_dir, converter.repo_name, file_name)
    assert converter.prepare_output_file(file_path) == expected_output
    assert os.path.exists(os.path.dirname(expected_output))
    shutil.rmtree(os.path.dirname(expected_output))

def test_yaml_to_txt(converter: RepoConverter):
    Repo.clone_from(converter.repo_url, converter.clone_dir)

    yaml_path = os.path.join(converter.clone_dir, 'folder_test', 'yaml_test.yaml')
    output_folder_path = os.path.join(converter.output_dir, converter.repo_name)
    output_file_path = os.path.join(output_folder_path, 'folder_test_yaml_test.txt')

    converter.yaml_to_txt(yaml_path)
    assert os.path.exists(output_file_path)
    f = open(output_file_path)
    assert f.read() == '# Unga Bunga!'
    f.close()
    shutil.rmtree(output_folder_path)
    shutil.rmtree(converter.clone_dir)

def test_python_to_txt(converter: RepoConverter):
    Repo.clone_from(converter.repo_url, converter.clone_dir)

    python_path = os.path.join(converter.clone_dir, 'folder_test', 'python_test.py')
    output_folder_path = os.path.join(converter.output_dir, converter.repo_name)
    output_file_path = os.path.join(output_folder_path, 'folder_test_python_test.txt')

    python_code = "def main():\n    print(\"Ooga Booga!\")\n\nif __name__ == \"__main__\":\n    main()"

    converter.py_to_txt(python_path)
    assert os.path.exists(output_file_path)
    f = open(output_file_path)
    assert f.read() == python_code
    f.close()
    shutil.rmtree(output_folder_path)
    shutil.rmtree(converter.clone_dir)

def test_md_to_txt(converter: RepoConverter):
    Repo.clone_from(converter.repo_url, converter.clone_dir)

    md_path = os.path.join(converter.clone_dir, 'md_test.md')
    output_folder_path = os.path.join(converter.output_dir, converter.repo_name)
    output_file_path = os.path.join(output_folder_path, 'md_test.txt')

    md_code = "# test_repository\nHello World!"

    converter.py_to_txt(md_path)
    assert os.path.exists(output_file_path)
    f = open(output_file_path)
    assert f.read() == md_code
    f.close()
    shutil.rmtree(output_folder_path)
    shutil.rmtree(converter.clone_dir)

def test_convert_repo(converter: RepoConverter):
    assert 'examples/end-to-end-examples/support_chatbot/tests/test_data/test_repository' == converter.convert_repo()
    assert os.path.exists('examples/end-to-end-examples/support_chatbot/tests/test_data/test_repository/folder_test_python_test.txt')
    assert os.path.exists('examples/end-to-end-examples/support_chatbot/tests/test_data/test_repository/folder_test_yaml_test.txt')
    assert os.path.exists('examples/end-to-end-examples/support_chatbot/tests/test_data/test_repository/md_test.txt')
    shutil.rmtree('examples/end-to-end-examples/support_chatbot/tests/test_data/test_repository')

