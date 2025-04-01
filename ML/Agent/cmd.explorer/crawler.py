
import subprocess
import sys
import time
import openai
import os
from dotenv import load_dotenv

load_dotenv()

llm_model = "gpt-4o-mini"
llm = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set the DEBUG environment variable to 1 to enable debug tracing
#os.environ["VERBOSE"] = "1"
os.environ["DEBUG"] = "1"
os.environ["WARNING"] = "1"


def verbose_trace(message):
    """
    Print a message to the console if the debug flag is set
    """
    if os.getenv("VERBOSE"):
        print(message)


def debug_trace(message):
    """
    Print a message to the console if the debug flag is set
    """
    if os.getenv("DEBUG"):
        print(message)


def warning_trace(message):
    """
    Print a message to the console if the debug flag is set
    """
    if os.getenv("WARNING"):
        print(message)


def llm_check_is_valid_help(help_output, llm_model=llm_model, llm=llm):
    """
    Use an LLM to check if the given output is a valid help output
    """
    llm_prompt = f"""
    You are a helpful assistant that extracts the help output of an executable.
    If there is no help output, return 'NO HELP'.
    If there is valid help output, return the help output in json format.
    The json format should be like this:
    {{
        "help": "help output",
        "ways_to_get_more_help": [
            ["functionality1 ", "parameter1", "parameter2", ...],
            ["functionality2", "parameter1", "parameter2", ...],
            ...
        ]
    }}
    {help_output}
    """

    try:
        verbose_trace(f"LLM prompt: {llm_prompt}")

        response = llm.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": llm_prompt}],
        )

        debug_trace(f"LLM response: {response.choices[0].message.content}")

        if response.choices[0].message.content == 'NO HELP':
            return False
        else:
            return response.choices[0].message.content
    except Exception as e:
        warning_trace(f"Error running LLM: {e}")
        return False


def run_executable(executable, parameters, timeout=2):
    """
    Launch an executable, get it's console output

    Args:
        executable (str): The path to the executable
        parameters (list): The parameters to pass to the executable
        timeout (int): The timeout for the executable
    returns: (output, error|None)
        output: The console output of the executable
        error: The error output of the executable
    """
    try:
        if len(parameters) == 1 and parameters[0] == "":
            parameters = []
        command = [executable] + parameters
        verbose_trace(f"Command: {command}")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        start_time = time.time()
        while process.poll() is None:
            if time.time() - start_time > timeout:
                process.terminate()
                warning_trace(f"Timeout: {command}")
                return None, "Timeout"
            time.sleep(0.1)
        output = process.stdout.read().decode('utf-8')
        debug_trace(f"Output: {command} : {output}")
        return output, None
    except Exception as e:
        warning_trace(f"Error: {command} : {e}")   
        return None, str(e) 


def get_help(executable, parameters):
    """
    Get help from an executable by launching it with different 'help' parameters

    Args:
        executable (str): The path to the executable
        parameters (list): The parameters to pass to the executable

    returns: (output, error|None)
        output: The help output of the executable
        error: The error output of the executable
    """
    
    help_parameters = ["", "-h", "--help", "/?", "/help"]
    for parameter in help_parameters:
        output, error = run_executable(executable, [parameter] + parameters)
        if output:
            yield output, None
    return None, "No help found"


def crawl_executable(executable, parameters):
    """
    Crawl an executable by launching it with different parameters

    Args:
        executable (str): The path to the executable
        parameters (list): The parameters to pass to the executable
    """

    errors = []
    for (help_output, help_error) in get_help(executable, parameters):

        if help_error:
            errors.append(help_error)
            continue

        if help_output:
            if llm_check_is_valid_help(help_output):
                return help_output, None
            else:
                errors.append(help_output)

    return None, errors



# main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=str)
    parser.add_argument("parameters", type=str, nargs="*")
    args = parser.parse_args()
    result, errors = crawl_executable(args.executable, args.parameters)
    if result:
        print(result)
    else:
        print(errors)