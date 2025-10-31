# Prompts
instructions = ("<repo>{path_to_repo}</repo>\n",
        "I have a C/C++ repository in the directory above.\n",
        "Can you identify whether the following function implemented in the repository is vulnerable or not?\n",
        "Your task is to analyze the function in the context of the repository in the commit {commit_hash} and provide a detailed explanation of your reasoning.\n",
        "Provide your answer in the following format:\n",
        "'''Vulnerable: <Yes/No>\n",
        "Explanation: <your analysis>'''\n",
        "Here is the function:\n",
        "{function_source_code}\n"
        "This function is in the commit {commit_hash}, then do NOT analyze the code after this point and do NOT search for any security report after the time of this commit date.\n",
        )