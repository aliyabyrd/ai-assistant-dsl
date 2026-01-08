"""
Aliya Byrd — Final Project
Voice-to-Action DSL

Basically: I type (or speak) a command, the program tokenizes it, parses it,
builds an AST, and then runs the action (graph / class code / docker command).

Extra credit I added:
- synonyms (plot/draw -> graph, build -> create, etc.)
- fixes math like 2x^3 -> 2*x^3
- optional voice mode (only works if speech packages are installed)
- saves graphs to a PNG automatically
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Optional
import re
import os
import time

import numpy as np
import matplotlib.pyplot as plt



'''EXTRA CREDIT #1: SYNONYMS / NORMALIZATION
This lets your "voice-to-action" input be more natural.
So for example: 
- "plot" / "draw" / "show" -> "graph"
- make" / "build" -> "create"
- "start" / "launch" -> "spin"
Also normalizes some object words.'''

SYNONYMS = {
    # actions
    "plot": "graph",
    "draw": "graph",
    "show": "graph",
    "chart": "graph",

    "make": "create",
    "build": "create",
    "generate": "create",

    "start": "spin",
    "launch": "spin",
    "run": "spin",

    # objects / filler
    "docker": "docker",
    "image": "running",   # sometimes people say "running image postgres"
}

# optional filler words we can allow / ignore
FILLER_WORDS = {
    "please", "can", "you", "me", "for", "to", "a", "the", "an"
}


def normalize_text(user_text: str) -> str:
    """
    EXTRA CREDIT #1:
    Normalizes user input:
    - lowercases
    - replaces synonyms
    - removes some filler words safely
    """
    text = user_text.strip().lower()

    # Keep punctuation for math, but normalize spacing
    words = re.split(r"(\s+)", text)  # split but keep spaces
    out = []
    for w in words:
        if w.isspace():
            out.append(w)
            continue

        # strip non-math punctuation around words
        bare = re.sub(r"^[^\w]+|[^\w]+$", "", w)

        if bare in FILLER_WORDS:
            # drop filler (preserve spacing by putting empty string)
            out.append("")
            continue

        # replace synonyms if exact word matches
        out.append(SYNONYMS.get(bare, w))

    normalized = "".join(out)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# =========================
#  EXTRA CREDIT #2: SMART MATH PREPROCESSING (2x -> 2*x)
# =========================
def preprocess_math(text: str) -> str:
    """
    EXTRA CREDIT #2:
    Converts common voice/math typing patterns into parser-friendly math.
    Example: "2x^3-4x^2+x-7" -> "2*x^3 - 4*x^2 + x - 7"
    """
    t = text

    # Insert * between number and identifier: 2x -> 2*x
    t = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", t)

    # Insert * between identifier and number? (rare) x2 -> x*2 (optional)
    t = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", t)

    # Make exponent with ^ safe: already handled by parser
    # Add spaces around operators for readability
    t = re.sub(r"([+\-*/^(),])", r" \1 ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t



#  Lexer (tokenizer)
class ParseError(Exception):
    """Custom exception for syntax / parsing errors."""
    pass


@dataclass
class Token:
    type: str
    value: str
    pos: int


KEYWORDS = {
    "graph", "the", "function", "between", "and",
    "create", "class", "with", "called", "a",
    "spin", "up", "container", "running", "docker"
}


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    text = text.strip()

    while i < len(text):
        c = text[i]

        if c.isspace():
            i += 1
            continue

        if c.isalpha():
            start = i
            while i < len(text) and text[i].isalpha():
                i += 1
            word = text[start:i].lower()
            ttype = "KEYWORD" if word in KEYWORDS else "IDENT"
            tokens.append(Token(ttype, word, start))
            continue

        if c.isdigit():
            start = i
            while i < len(text) and (text[i].isdigit() or text[i] == "."):
                i += 1
            tokens.append(Token("NUMBER", text[start:i], start))
            continue

        if c in "+-*/^(),":
            tokens.append(Token(c, c, i))
            i += 1
            continue

        raise ParseError(f"Unknown character '{c}' at position {i}")

    tokens.append(Token("EOF", "", len(text)))
    return tokens

#  AST Nodes

@dataclass
class Expr:
    pass

@dataclass
class Number(Expr):
    value: float

@dataclass
class Variable(Expr):
    name: str

@dataclass
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr

@dataclass
class UnaryOp(Expr):
    op: str
    operand: Expr


@dataclass
class GraphCommand:
    expr: Expr
    start: float
    end: float

@dataclass
class CreateClassCommand:
    name: str
    fields: List[str]

@dataclass
class SpinContainerCommand:
    image: str


Command = Union[GraphCommand, CreateClassCommand, SpinContainerCommand]

#  Parser - recursive descent

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def match(self, *types_or_vals) -> Token:
        tok = self.current()
        for t in types_or_vals:
            if tok.type == t or tok.value == t:
                self.pos += 1
                return tok
        expected = " or ".join(types_or_vals)
        raise ParseError(
            f"Expected {expected} at position {tok.pos}, got {tok.type}:{tok.value}"
        )

    def consume_optional(self, *vals) -> bool:
        if self.current().value in vals:
            self.pos += 1
            return True
        return False

    def parse(self) -> Command:
        tok = self.current()
        if tok.value == "graph":
            cmd = self.parse_graph_command()
        elif tok.value == "create":
            cmd = self.parse_create_class()
        elif tok.value == "spin":
            cmd = self.parse_spin_cmd()
        else:
            raise ParseError(
                f"Unknown command '{tok.value}' at position {tok.pos}. "
                f"Expected 'graph', 'create', or 'spin'."
            )

        if self.current().type != "EOF":
            tok = self.current()
            raise ParseError(f"Unexpected extra input at position {tok.pos}: {tok.value}")

        return cmd

    # expression grammar 
    def parse_expression(self) -> Expr:
        node = self.parse_term()
        while self.current().value in ("+", "-"):
            op = self.current().value
            self.pos += 1
            right = self.parse_term()
            node = BinaryOp(op, node, right)
        return node

    def parse_term(self) -> Expr:
        node = self.parse_factor()
        while self.current().value in ("*", "/"):
            op = self.current().value
            self.pos += 1
            right = self.parse_factor()
            node = BinaryOp(op, node, right)
        return node

    def parse_factor(self) -> Expr:
        if self.current().value in ("+", "-"):
            op = self.current().value
            self.pos += 1
            return UnaryOp(op, self.parse_factor())

        tok = self.current()
        if tok.type == "NUMBER":
            self.pos += 1
            node = Number(float(tok.value))
        elif tok.type == "IDENT":
            self.pos += 1
            node = Variable(tok.value)
        elif tok.value == "(":
            self.pos += 1
            node = self.parse_expression()
            self.match(")")
        else:
            raise ParseError(f"Unexpected token {tok.value} at {tok.pos}")

        # exponent is right-associative
        if self.current().value == "^":
            op = self.current().value
            self.pos += 1
            right = self.parse_factor()
            node = BinaryOp(op, node, right)

        return node

    def parse_signed_number(self) -> float:
        sign = 1
        if self.current().value == "-":
            sign = -1
            self.pos += 1
        tok = self.current()
        if tok.type != "NUMBER":
            raise ParseError(f"Expected number at position {tok.pos}")
        self.pos += 1
        return sign * float(tok.value)

    def parse_range(self):
        self.match("between")
        start = self.parse_signed_number()
        self.match("and")
        end = self.parse_signed_number()
        return start, end

    # command parsers
    def parse_graph_command(self) -> GraphCommand:
        self.match("graph")
        self.consume_optional("the")
        self.consume_optional("function")
        expr = self.parse_expression()
        start, end = self.parse_range()
        return GraphCommand(expr, start, end)

    def parse_create_class(self) -> CreateClassCommand:
        self.match("create")
        self.consume_optional("a")
        self.match("class")
        self.consume_optional("called")
        name = self.match("IDENT").value

        self.match("with")
        fields = []
        while True:
            fields.append(self.match("IDENT").value)
            if self.current().value == ",":
                self.pos += 1
            else:
                break
        return CreateClassCommand(name, fields)

    def parse_spin_cmd(self) -> SpinContainerCommand:
        self.match("spin")
        self.match("up")
        self.consume_optional("a")
        self.consume_optional("docker")
        self.match("container")
        self.match("running")
        image = self.match("IDENT").value
        return SpinContainerCommand(image)



#  AST printing (show in my demo)

def ast_to_text(cmd: Command) -> str:
    if isinstance(cmd, GraphCommand):
        return (
            "GraphCommand\n"
            "├── expr: " + expr_to_text(cmd.expr) + "\n"
            f"└── range: start={cmd.start}, end={cmd.end}"
        )
    if isinstance(cmd, CreateClassCommand):
        return (
            "CreateClassCommand\n"
            f"├── name: {cmd.name}\n"
            f"└── fields: {cmd.fields}"
        )
    if isinstance(cmd, SpinContainerCommand):
        return (
            "SpinContainerCommand\n"
            f"└── image: {cmd.image}"
        )
    return "UnknownCommand"

def expr_to_text(expr: Expr) -> str:
    if isinstance(expr, Number):
        return f"Number({expr.value})"
    if isinstance(expr, Variable):
        return f'Variable("{expr.name}")'
    if isinstance(expr, UnaryOp):
        return f'UnaryOp("{expr.op}", {expr_to_text(expr.operand)})'
    if isinstance(expr, BinaryOp):
        return f'BinaryOp("{expr.op}", {expr_to_text(expr.left)}, {expr_to_text(expr.right)})'
    return "Expr(?)"


#  Executor
def eval_expr(expr: Expr, x):
    if isinstance(expr, Number):
        return expr.value
    if isinstance(expr, Variable):
        if expr.name == "x":
            return x
        raise ValueError(f"Unknown variable {expr.name}")
    if isinstance(expr, UnaryOp):
        val = eval_expr(expr.operand, x)
        return -val if expr.op == "-" else val
    if isinstance(expr, BinaryOp):
        left = eval_expr(expr.left, x)
        right = eval_expr(expr.right, x)
        if expr.op == "+": return left + right
        if expr.op == "-": return left - right
        if expr.op == "*": return left * right
        if expr.op == "/": return left / right
        if expr.op == "^": return left ** right
    raise ValueError("Invalid expression")


def execute_command(cmd: Command):
    if isinstance(cmd, GraphCommand):
        xs = np.linspace(cmd.start, cmd.end, 400)
        ys = eval_expr(cmd.expr, xs)

        plt.figure()
        plt.plot(xs, ys)
        plt.title("Graph of Function")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)

        
        #  EXTRA CREDIT #4: Saving graph to png
    
        out_dir = "demo_outputs"
        os.makedirs(out_dir, exist_ok=True)
        filename = f"graph_{int(time.time())}.png"
        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=150)

        plt.show()
        return f"Graph generated. (Saved to {path})"

    if isinstance(cmd, CreateClassCommand):
        print("\nGenerated Python Class:\n")
        print(f"class {cmd.name.capitalize()}:")
        print(f"    def __init__(self, {', '.join(cmd.fields)}):")
        for f in cmd.fields:
            print(f"        self.{f} = {f}")
        return "Class generated."

    if isinstance(cmd, SpinContainerCommand):
        print("\nDocker Command:\n")
        print(f"docker run -d {cmd.image}")
        return "Docker command generated."

    return "Unknown command."



# EXTRA CREDIT #3: optional speech-to-text 

def try_speech_to_text() -> Optional[str]:
    """
    If SpeechRecognition + PyAudio are installed and mic access works,
    this records 1 short command and converts it to text.
    If not installed, returns None (so the program still works).
    """
    try:
        import speech_recognition as sr  # type: ignore
    except Exception:
        return None

    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🎙️ Listening... (say one command)")
            audio = r.listen(source, phrase_time_limit=6)
        text = r.recognize_google(audio)
        return text
    except Exception:
        return None

#  REPL (run loop)

def repl():
    print("Voice-to-Action AI DSL (Extra Credit)")
    print("Type a command, or type 'quit' to exit.")
    print("Tip: You can also type: voice  (to try speech-to-text if installed)\n")

    while True:
        raw = input(">> ").strip()
        if raw.lower() in {"quit", "exit"}:
            break

        # If user wants voice mode:
        if raw.lower() == "voice":
            spoken = try_speech_to_text()
            if spoken is None:
                print("Voice mode not available (install speechrecognition + pyaudio). Falling back to typing.\n")
                continue
            print(f'Heard: "{spoken}"')
            raw = spoken

        # EXTRA CREDIT normalization + math preprocessing
        normalized = normalize_text(raw)
        normalized = preprocess_math(normalized)

        try:
            tokens = tokenize(normalized)
            parser = Parser(tokens)
            cmd = parser.parse()

            print("\nAST:")
            print(ast_to_text(cmd))

            result = execute_command(cmd)
            print("\n" + result)
            print("-------------------------\n")

        except Exception as e:
            print(f"ERROR: {e}\n")


if __name__ == "__main__":
    repl()
