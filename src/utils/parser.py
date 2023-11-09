def binary_parser(output):
    sections = output.split('\n')
    analysis = ""
    final_answer = ""
    for section in sections:
        section_lower = section.lower()
        if section_lower.startswith("analysis: "):
            analysis = section_lower.replace("analysis: ", "")
        elif section_lower.startswith("final answer: "):
            final_answer = section_lower.replace("final answer: ", "").strip()
    # Ensuring that we only get 'yes' or 'no'
    if 'yes' in final_answer:
        final_answer = 'yes'
    elif 'no' in final_answer:
        final_answer = 'no'
    else:
        final_answer = ""
    return analysis, final_answer


def mc_parser(output):
    sections = output.split('\n')
    analysis = ""
    final_answer = ""
    for section in sections:
        section_lower = section.lower()
        if section_lower.startswith("analysis: "):
            analysis = section_lower.replace("analysis: ", "")
        elif section_lower.startswith("final answer: "):
            final_answer = section_lower.replace("final answer: ", "").strip()
    # We only keep the first character of the final answer which should be the letter representing the chosen option
    final_answer = final_answer[0] if final_answer and final_answer[0].isalpha() else ""
    return analysis, final_answer


def text_parser(output):
    sections = output.split('\n')
    analysis = ""
    final_answer = ""
    for section in sections:
        section_lower = section.lower()
        if section_lower.startswith("analysis: "):
            analysis = section_lower.replace("analysis: ", "")
        elif section_lower.startswith("final answer: "):
            final_answer = section_lower.replace("final answer: ", "").strip()
    return analysis, final_answer
