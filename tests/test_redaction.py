"""Redaction: secrets in every shape a setting is written, and no text that takes long."""

import json
import time

import pytest

from llmshift.redact import Redactor

AWS = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
SYM = "xK9#mP2&vL5?qR8(wN3)"  # a generated password: & ? ( ) are part of it


def test_hostile_text_takes_milliseconds_not_seconds():
    hostile = [
        ("-----BEGIN " + "PRIVATE KEY-----\n") * 8000,  # 224 KB of BEGINs with no END held the GIL for 15 s
        ("-----BEGIN " + "RSA PRIVATE KEY-----AAAA\n") * 6000,
        ("x_" * 20 + "passwor") * 5000,  # a long name before the word, never finished
        "password=." * 20000,
        "pwd=a1!@#$%^&*()" * 15000,
        "token" * 40000,
    ]
    for text in hostile:
        start = time.perf_counter()
        Redactor(pii=True).text(text)
        assert time.perf_counter() - start < 2, text[:40]  # linear: about 0.1 s on a laptop


@pytest.mark.parametrize("text", [
    f"password={SYM}",
    f"password: {SYM}",
    f'"password": "{SYM}"',
    f"password = '{SYM}'",
    f"login with password: {SYM} now",
    f"export DB_PASSWORD={SYM}",
    f"<password>{SYM}</password>",
    json.dumps({"name": "login", "arguments": json.dumps({"user": "bob", "password": SYM})}),  # a tool call's arguments
])
def test_a_password_with_symbols_is_redacted_whole(text):
    out = Redactor().text(text)
    assert "mP2&vL5" not in out and "qR8(wN3" not in out and "<SECRET_1>" in out, out


@pytest.mark.parametrize("text, expected", [
    ("login(password=pw12345)", "login(password=<SECRET_1>)"),
    ("Server=db;User Id=sa;Password=Hu7nter22;Database=app", "Server=db;User Id=sa;Password=<SECRET_1>;Database=app"),
    ("?api_key=abc123def&format=json", "?api_key=<SECRET_1>&format=json"),
    ("Your password: Hu7nter22.", "Your password: <SECRET_1>."),
    ("api_key=abc123def; next", "api_key=<SECRET_1>; next"),
    ("my_really_long_service_name_for_the_main_database_password=Hu7nter22", "my_really_long_service_name_for_the_main_database_password=<SECRET_1>"),
    ('password = "It\'s2Secret!"', 'password = "<SECRET_1>"'),
    ("token=abc123,password: Hu7nter22", "token=<SECRET_1>,password: <SECRET_2>"),
])
def test_only_the_value_is_replaced(text, expected):
    assert Redactor().text(text) == expected


def test_a_long_token_is_redacted_to_its_end():
    token = "ya29.a0Af" + "x1" * 300  # 609 characters: the old rules stopped at 256
    for text in (f"access_token={token}", json.dumps({"access_token": token}), json.dumps(json.dumps({"access_token": token}))):
        out = Redactor().text(text)
        assert "x1x1" not in out, out[:80]
    assert "x1x1" not in json.dumps(Redactor().obj({"access_token": token}))


@pytest.mark.parametrize("text", [
    f"key={AWS}",
    f'"key": "{AWS}"',
    f"Key: {AWS}",
    f"aws_secret_access_key = {AWS}",
])
def test_an_aws_secret_key_after_a_plain_key_is_redacted(text):
    assert AWS not in Redactor().text(text)


@pytest.mark.parametrize("text", [
    "cache key=3f786850e387550fdab836ed7e6dc881de23001b",  # a SHA-1, lower-case hex
    "token=123456&format=json",
    "token = tokens[0]",
    "password = hashlib.sha256(pw1)",
    "api_key = os.environ['OPENAI_API_KEY_2']",
    "api_key=st.secrets['openai_key_2']",
    "password: ${DB_PASSWORD_1}",
    'password: "${DB_PASSWORD_1}"',
    "token=${{ secrets.GITHUB_TOKEN_2 }}",
    'password: "<SECRET_1>", token=<SECRET_2>',  # already redacted: a second pass keeps it
    "PWD=/home/user1 OLDPWD=/home/user1/app",  # the working directory, not a password
    "<maxTokens>4096</maxTokens>",
    "the password is Hu7nter22, keep it safe",  # prose with no = or : is not a setting
])
def test_code_placeholders_and_prose_are_left_alone(text):
    assert Redactor().text(text) == text


def test_json_values_under_a_secret_name_are_redacted_whatever_they_hold():
    data = {
        "input": {"username": "bob", "password": SYM},
        "login": {"password": "correct horse battery staple"},
        "headers": {"x-api-key": "0123456789abcdef0123456789abcdef"},
        "aws": {"key": AWS},
        "basic": {"credentials": "user:pa55word"},
    }
    out = json.dumps(Redactor().obj(data))
    for secret in (SYM, "correct horse", "0123456789abcdef", AWS, "pa55word"):
        assert secret not in out
    assert '"username": "bob"' in out


def test_json_values_that_only_name_a_secret_are_kept():
    data = {
        "logprobs": {"content": [{"token": "Hello123", "logprob": -0.1, "top_logprobs": [{"token": "World42", "logprob": -2.3}]}]},
        "logprobsResult": {"chosenCandidates": [{"token": "Hello123", "logProbability": -0.1}]},  # Gemini
        "form": {"password": "Password", "secret": "<SECRET_1>", "token": "${TOKEN}", "password_hint": "please enter it"},
        "shell": {"pwd": "/workspace/app"},
        "flags": {"is_secret": "false", "max_tokens": "abc123x", "eos_token": "<|endoftext|>"},
        "schema": {"properties": {"password": {"type": "string"}}},
        "s3": {"key": "uploads/2024/photo123.jpg"},
    }
    assert Redactor().obj(data) == data
