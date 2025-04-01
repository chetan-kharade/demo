import streamlit as st
import sys
import io

def execute_python_code(code):
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        exec(code, {})
        output = redirected_output.getvalue()
    except Exception as e:
        output = str(e)
    finally:
        sys.stdout = old_stdout

    return output

def main():
    st.title("Python Compiler Chatbot 🐍")
    st.write("Type your Python code and get the output.")

    user_input = st.text_area("Enter your Python code here:")
    if st.button("Run Code"):
        if user_input.strip():
            output = execute_python_code(user_input)
            st.code(output, language="python")
        else:
            st.warning("Please enter some Python code.")

if __name__ == "__main__":
    main()
