import gradio as gr
import json
import logging

logging.basicConfig(level=logging.INFO)
import re
from datetime import datetime
from backend import initialize as initialize_backend
from research_agent import ResearchAgent
from state import Dialog, Role


initialize_backend()


dialog = Dialog(limit=5)
agent = ResearchAgent(
    {
        "dialog": dialog,
        "tool_call_iterations": 0,
    }
)


def chat(message, history):
    """Return a full response (non-streaming)."""
    if not message.strip():
        yield "Please enter a message."

    # yield "This is a placeholder response: {}".format(message)

    dialog.add_message(role=Role.USER, content=message)
    agent.state["tool_call_iterations"] = 0  # reset tool call budge
    for chunk in agent.run():
        yield chunk


def _is_tabular_data(data: dict) -> bool:
    """
    Determine if JSON data represents tabular data.
    Returns True if it's a dict with list values of equal length (>=1).
    """
    if not isinstance(data, dict):
        return False

    # Must have at least 2 keys to be worth displaying as a table
    if len(data) < 2:
        return False

    # All values must be lists
    if not all(isinstance(v, list) for v in data.values()):
        return False

    # Get all list lengths
    lengths = [len(v) for v in data.values()]

    # All lists must have the same length and length >= 1
    if not lengths:
        return False

    # All lists must be same length
    if len(set(lengths)) != 1:
        return False

    # Must have at least one element
    return lengths[0] >= 1


def _format_value(val) -> str:
    """Format a value for table display."""
    if val is None:
        return ""
    elif isinstance(val, float):
        # Format floats nicely
        if abs(val) >= 1000000:
            return f"{val:,.2f}"
        elif abs(val) >= 1:
            return f"{val:.4f}".rstrip("0").rstrip(".")
        else:
            return f"{val:.6f}".rstrip("0").rstrip(".")
    elif isinstance(val, (int, bool)):
        return str(val)
    else:
        return str(val)


def _json_to_markdown_table(data: dict) -> str:
    """
    Convert tabular JSON data to a markdown table.
    Assumes data is a dict with equal-length lists as values.
    """
    if not _is_tabular_data(data):
        return None

    # Get column names and data
    columns = list(data.keys())
    num_rows = len(data[columns[0]])

    # Build table
    lines = []

    # Header row
    header = "| " + " | ".join(columns) + " |"
    lines.append(header)

    # Separator row
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.append(separator)

    # Data rows
    for i in range(num_rows):
        row_values = [_format_value(data[col][i]) for col in columns]
        row = "| " + " | ".join(row_values) + " |"
        lines.append(row)

    return "\n".join(lines)


def _format_json(json_str: str) -> tuple[str, bool]:
    """
    Try to parse and format a JSON string.
    Returns (formatted_string, is_valid) tuple.
    If data is tabular, formats as a markdown table.
    """
    try:
        parsed = json.loads(json_str)

        # Check if it's tabular data
        if _is_tabular_data(parsed):
            table = _json_to_markdown_table(parsed)
            if table:
                return table, True

        # Otherwise, format as regular JSON
        formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
        return f"```json\n{formatted}\n```", True
    except json.JSONDecodeError:
        return json_str, False
    except Exception as e:
        # If table formatting fails, fall back to JSON
        try:
            parsed = json.loads(json_str)
            formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
            return f"```json\n{formatted}\n```", True
        except:
            return json_str, False


def _is_python_code(text: str) -> bool:
    """
    Detect if text is Python code using specific patterns.
    Must be very conservative to avoid false positives with JSON.
    """
    if not text or len(text.strip()) < 10:
        return False

    # First, check if it looks like JSON - if so, it's NOT Python
    text_stripped = text.strip()
    if text_stripped.startswith("{") and text_stripped.endswith("}"):
        try:
            json.loads(text_stripped)
            return False  # Valid JSON, not Python
        except json.JSONDecodeError:
            pass  # Not valid JSON, continue checking

    text_lower = text.lower()

    # Strong Python indicators (must have at least one)
    strong_patterns = [
        r"\bdef\s+\w+\s*\(",  # Function definitions
        r"\bclass\s+\w+",  # Class definitions
        r"\bimport\s+\w+",  # Import statements
        r"\bfrom\s+\w+\s+import",  # From imports
    ]

    has_strong_pattern = any(re.search(p, text, re.MULTILINE) for p in strong_patterns)

    # Without a strong pattern, require very clear indicators
    if not has_strong_pattern:
        # Need multiple clear Python-only keywords
        python_only_keywords = [
            "def ",
            "class ",
            "import ",
            "from ",
            "__init__",
            "self.",
        ]
        keyword_count = sum(1 for kw in python_only_keywords if kw in text_lower)

        has_proper_indentation = bool(re.search(r"^\s{4,}", text, re.MULTILINE))
        is_multiline = text.count("\n") >= 3

        return keyword_count >= 2 and has_proper_indentation and is_multiline

    return True


def _has_markdown_fences(text: str) -> bool:
    """Check if text already contains markdown code fences."""
    return bool(re.search(r"```\w*\n", text))


def _filter_nested_blocks(blocks: list) -> list:
    """
    Filter out nested blocks, keeping only outermost ones.
    Blocks should be tuples of (start_pos, end_pos, ...).
    """
    if not blocks:
        return []

    filtered = []
    sorted_blocks = sorted(
        blocks, key=lambda x: (x[0], -x[1])
    )  # Sort by start, then by end descending

    for block in sorted_blocks:
        start, end = block[0], block[1]
        # Check if this block is nested within any already filtered block
        is_nested = any(
            start >= prev_start and end <= prev_end
            for prev_start, prev_end, *_ in filtered
        )
        if not is_nested:
            filtered.append(block)

    return filtered


def _extract_json_blocks(text: str) -> list:
    """
    Robust single-pass scanner that returns outermost JSON blocks found anywhere in text.
    Each returned tuple is (start_pos, end_pos, json_string).
    It respects quoted strings (with escapes) and validates blocks with json.loads().
    """
    blocks = []
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]
        # Look for an opening brace that might start a JSON object
        if ch != "{":
            i += 1
            continue

        start = i
        in_string = False
        escape_next = False
        brace_count = 0
        found = False

        for j in range(i, n):
            c = text[j]

            if escape_next:
                escape_next = False
                continue

            if c == "\\":
                escape_next = True
                continue

            if c == '"':
                in_string = not in_string
                continue

            # only count braces when not inside a JSON string
            if in_string:
                continue

            if c == "{":
                brace_count += 1
            elif c == "}":
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[start : j + 1]
                    # validate it's real JSON
                    try:
                        json.loads(candidate)
                        blocks.append((start, j + 1, candidate))
                        found = True
                        i = j  # advance outer loop to after this block
                    except json.JSONDecodeError:
                        # not valid JSON, ignore this candidate and continue scanning
                        pass
                    break

        # If we didn't find a matching end, break (partial JSON - let streaming handle it)
        if not found:
            # stop scanning so that partial JSON at end is left for future streamed chunks
            break

        i += 1

    return blocks


def _clean_and_format_text(text: str) -> str:
    """
    Clean and format text from streamed responses.
    Handles: JSON, Python code, tables, and plain text.
    """
    if not text:
        return ""

    # Remove "助手： " prefix
    text = re.sub(r"^助手[：:]\s*", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()

    # If already has markdown fences, return as-is
    if _has_markdown_fences(text):
        return text

    # Try parsing as pure JSON first (most common case)
    try:
        parsed = json.loads(text)
        # Check if it's tabular data
        if _is_tabular_data(parsed):
            table = _json_to_markdown_table(parsed)
            if table:
                return table
        # Otherwise format as JSON
        return f"```json\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n```"
    except json.JSONDecodeError:
        pass
    except Exception:
        pass

        # Find JSON blocks using the unified, robust scanner
    json_blocks = _extract_json_blocks(text)

    # No JSON found - check if it's Python code or plain text
    if not json_blocks:
        if _is_python_code(text):
            return f"```python\n{text}\n```"
        return text

    # Filter nested JSON blocks (keep only outermost)
    json_blocks = _filter_nested_blocks(json_blocks)

    # Remove duplicates (blocks with same start position)
    seen_starts = set()
    unique_blocks = []
    for block in json_blocks:
        if block[0] not in seen_starts:
            unique_blocks.append(block)
            seen_starts.add(block[0])

    # Sort by start position
    unique_blocks.sort(key=lambda x: x[0])

    # No JSON found - check if it's Python code or plain text
    if not unique_blocks:
        if _is_python_code(text):
            return f"```python\n{text}\n```"
        return text

    # Filter nested JSON blocks
    unique_blocks = _filter_nested_blocks(unique_blocks)

    # Build result with formatted JSON/tables and text segments
    result_parts = []
    current_pos = 0

    for start, end, json_str in unique_blocks:
        # Process text before this JSON block
        if start > current_pos:
            pre_text = text[current_pos:start].strip()
            if pre_text:
                if _is_python_code(pre_text):
                    result_parts.append(f"```python\n{pre_text}\n```")
                else:
                    result_parts.append(pre_text)

        # Format and add JSON block (or table)
        formatted, _ = _format_json(json_str)
        result_parts.append(formatted)
        current_pos = end

    # Process remaining text after last JSON block
    if current_pos < len(text):
        remaining = text[current_pos:].strip()
        if remaining:
            if _is_python_code(remaining):
                result_parts.append(f"```python\n{remaining}\n```")
            else:
                result_parts.append(remaining)

    return "\n\n".join(result_parts)


def respond(message, chat_history):
    if not message.strip():
        chat_history.append((message, "Please enter a message."))
        yield "", chat_history
        return

    chat_history.append((message, ""))
    yield "", chat_history

    formatted_segments = []  # store already formatted parts
    buffer = ""  # accumulate partial unformatted text
    prev_chunk = ""

    for chunk in chat(message, chat_history):
        if chunk.startswith(prev_chunk):
            new_text = chunk[len(prev_chunk) :]
        else:
            new_text = chunk  # fallback: treat as delta

        prev_chunk = chunk
        buffer += new_text

        # Try to detect any *complete* JSON or Python block in the new buffer
        json_blocks = _extract_json_blocks(buffer)

        if json_blocks:
            # Take only outermost completed block(s)
            json_blocks = _filter_nested_blocks(json_blocks)
            end_pos = json_blocks[-1][1]

            # Format up to last completed block
            formatted_part = _clean_and_format_text(buffer[:end_pos])
            formatted_segments.append(formatted_part)

            # Keep remainder in buffer
            buffer = buffer[end_pos:]
        else:
            # No complete block yet → don't reformat previous text
            pass

        # Display concatenated formatted segments + current buffer (unformatted)
        display = "\n\n".join(formatted_segments + [buffer])
        chat_history[-1] = (message, display)
        yield "", chat_history

    # After stream ends, format any remaining partial buffer
    if buffer.strip():
        formatted_segments.append(_clean_and_format_text(buffer))

    chat_history[-1] = (message, "\n\n".join(formatted_segments))
    yield "", chat_history


def clear_chat():
    """Clear the chat UI and reset internal dialog state."""
    # AGENT.dialog.clear()
    return []


def capture_and_save_screenshot(chat_history):
    """Save text file and return timestamp for screenshot capture."""
    if not chat_history:
        return "⚠️ No chat history to save.", None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = f"chat_screenshot_{timestamp}.txt"

    try:
        # Save text file
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("CHAT HISTORY SCREENSHOT\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for i, (user_msg, bot_msg) in enumerate(chat_history, 1):
                user_msg = str(user_msg) if user_msg else ""
                bot_msg = str(bot_msg) if bot_msg else ""
                f.write(f"[Message {i}]\n")
                f.write(f"User: {user_msg}\n")
                f.write(f"\nAssistant: {bot_msg}\n")
                f.write("\n" + "-" * 80 + "\n\n")

        print(f"Text file saved: {txt_filename}")
        return f"✅ Text saved: {txt_filename}\n", timestamp

    except Exception as e:
        import traceback

        print(f"Error saving text: {traceback.format_exc()}")
        return f"❌ Error: {str(e)}", None


# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    table {
        background-color: #fffef8 !important;
    }
    table th {
        background-color: #fffef8 !important;
    }
    table td {
        background-color: #fffef8 !important;
    }
    """,
) as demo:

    gr.Markdown("# 🤖 Demo")

    chatbot = gr.Chatbot(
        height=900,
        show_label=False,
        avatar_images=[
            "https://api.dicebear.com/7.x/avataaars/svg?seed=user",
            "https://api.dicebear.com/7.x/bottts/svg?seed=bot",
        ],
        elem_id="chatbot-container",
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            scale=4,
            container=False,
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")

    # Hidden components for screenshot workflow
    timestamp_state = gr.State()
    screenshot_data_state = gr.State()

    status_text = gr.Textbox(
        label="Status",
        show_label=False,
        interactive=False,
        lines=1,
        max_lines=1,
        autoscroll=False,
    )

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Chat")
        save_txt_btn = gr.Button("📝 Save Text Log")
        save_txt_btn.click(
            capture_and_save_screenshot,
            inputs=[chatbot],
            outputs=[status_text, timestamp_state],
        )

    gr.Examples(
        examples=[
            "Why is the sky blue?",
            "Hello, can you help me with my research?",
            "Summarize the latest research on quantum computing.",
            "why is grass green? and how tall is mount everest?",
            "Write a paper to discuss the influence of AI interaction on interpersonal relations, considering AI's potential to fundamentally change how and why individuals relate to each other.",
        ],
        inputs=msg,
        label="Try these examples:",
    )

    # Event handlers
    submit_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear_btn.click(clear_chat, None, chatbot, queue=False)


if __name__ == "__main__":
    demo.launch(
        share=False,
        debug=False,
        server_name="127.0.0.1",
        server_port=8000,
        inbrowser=True,
        show_error=True,
    )
