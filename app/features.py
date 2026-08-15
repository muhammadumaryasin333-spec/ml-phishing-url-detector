"""Approximate live extraction of the PhiUSIIL URL and webpage feature schema."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
import ipaddress
import math
import re
from urllib.parse import unquote, urljoin, urlsplit

import pandas as pd

from .fetcher import FetchResult


AUDITED_FEATURE_NAMES = [
    "URLLength", "DomainLength", "IsDomainIP", "TLD", "TLDLength",
    "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS", "LineOfCode",
    "LargestLineLength", "HasTitle", "DomainTitleMatchScore", "URLTitleMatchScore",
    "HasFavicon", "Robots", "IsResponsive", "NoOfURLRedirect", "NoOfSelfRedirect",
    "HasDescription", "NoOfPopup", "NoOfiFrame", "HasExternalFormSubmit",
    "HasSocialNet", "HasSubmitButton", "HasHiddenFields", "HasPasswordField",
    "Bank", "Pay", "Crypto", "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS",
    "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef",
]
SOCIAL_HOSTS = {
    "facebook.com", "instagram.com", "linkedin.com", "pinterest.com", "tiktok.com",
    "twitter.com", "x.com", "youtube.com",
}
BANK_TERMS = {"bank", "banking", "account", "credit", "debit", "iban"}
PAY_TERMS = {"pay", "payment", "paypal", "checkout", "billing", "invoice"}
CRYPTO_TERMS = {"bitcoin", "crypto", "ethereum", "wallet", "blockchain", "usdt"}
@dataclass(frozen=True)
class FeatureExtractionResult:
    """One complete model row plus disclosed approximation metadata."""

    audited: pd.DataFrame
    coverage: float
    approximate_features: tuple[str, ...]
    final_url_changed: bool


def _attributes(attributes: list[tuple[str, str | None]]) -> dict[str, str]:
    return {key.casefold(): (value or "") for key, value in attributes}


class _PageParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.in_title = False
        self.links: list[str] = []
        self.form_actions: list[str] = []
        self.meta_refreshes: list[str] = []
        self.has_favicon = False
        self.has_robots = False
        self.is_responsive = False
        self.has_description = False
        self.has_submit = False
        self.has_hidden = False
        self.has_password = False
        self.image_count = 0
        self.css_count = 0
        self.js_count = 0
        self.iframe_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        values = _attributes(attrs)
        if tag == "title":
            self.in_title = True
        elif tag == "a":
            self.links.append(values.get("href", ""))
        elif tag == "form":
            self.form_actions.append(values.get("action", ""))
        elif tag == "img":
            self.image_count += 1
        elif tag == "script":
            self.js_count += 1
        elif tag == "iframe":
            self.iframe_count += 1
        elif tag == "link":
            relation = values.get("rel", "").casefold().split()
            if "stylesheet" in relation:
                self.css_count += 1
            if any(item in {"icon", "shortcut", "shortcut-icon"} for item in relation):
                self.has_favicon = True
        elif tag == "meta":
            name = values.get("name", "").casefold()
            http_equiv = values.get("http-equiv", "").casefold()
            content = values.get("content", "")
            if name == "robots":
                self.has_robots = True
            if name == "viewport":
                self.is_responsive = True
            if name == "description" and content.strip():
                self.has_description = True
            if http_equiv == "refresh":
                self.meta_refreshes.append(content)
        elif tag in {"button", "input"}:
            input_type = values.get("type", "").casefold()
            if tag == "button" or input_type in {"submit", "image"}:
                self.has_submit = True
            if input_type == "hidden":
                self.has_hidden = True
            if input_type == "password":
                self.has_password = True

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)


def _normal_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", unquote(value).casefold()))


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return 100.0
    return round(100.0 * SequenceMatcher(None, left, right).ratio(), 6)


def _registrable_hint(hostname: str) -> str:
    labels = hostname.casefold().strip(".").split(".")
    return labels[-2] if len(labels) >= 2 else labels[0]


def _is_ip(hostname: str) -> int:
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def _is_external(reference: str, page_url: str) -> bool:
    resolved = urlsplit(urljoin(page_url, reference))
    page_host = (urlsplit(page_url).hostname or "").casefold()
    return bool(resolved.hostname and resolved.hostname.casefold() != page_host)


def _is_empty_reference(reference: str) -> bool:
    value = reference.strip().casefold()
    return not value or value == "#" or value.startswith(("javascript:", "mailto:", "tel:"))


def extract_features(submitted_url: str, fetched: FetchResult) -> FeatureExtractionResult:
    """Compute all saved tabular model inputs from bounded HTML and URL text.

    PhiUSIIL does not publish an executable extraction contract. Derived and
    semantic fields are therefore deliberately disclosed as approximations.
    """
    charset_match = re.search(rb"charset\s*=\s*['\"]?([a-zA-Z0-9._-]+)", fetched.body[:4096], re.I)
    charset = charset_match.group(1).decode("ascii", "ignore") if charset_match else "utf-8"
    try:
        html = fetched.body.decode(charset, "replace")
    except LookupError:
        html = fetched.body.decode("utf-8", "replace")

    parser = _PageParser(fetched.final_url)
    parser.feed(html)
    parsed = urlsplit(fetched.final_url)
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    tld = hostname.rsplit(".", 1)[-1] if "." in hostname else hostname
    url_without_scheme = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", fetched.final_url)
    url_without_www = re.sub(r"^www\.", "", url_without_scheme, flags=re.I)
    url_length = len(fetched.final_url)
    letters = sum(character.isalpha() for character in url_without_www)
    digits = sum(character.isdigit() for character in url_without_www)
    specials = sum(not character.isalnum() for character in url_without_www)
    percent_sequences = re.findall(r"%[0-9a-fA-F]{2}", fetched.final_url)
    title = " ".join(" ".join(parser.title_parts).split())
    domain_text = _normal_text(_registrable_hint(hostname))
    url_text = _normal_text(url_without_www)
    title_text = _normal_text(title)
    domain_title_score = _similarity(domain_text, title_text)
    url_title_score = _similarity(url_text, title_text)

    self_refs = empty_refs = external_refs = 0
    social = False
    for reference in parser.links:
        if _is_empty_reference(reference):
            empty_refs += 1
            continue
        resolved_host = (urlsplit(urljoin(fetched.final_url, reference)).hostname or "").casefold()
        if any(resolved_host == host or resolved_host.endswith("." + host) for host in SOCIAL_HOSTS):
            social = True
        if _is_external(reference, fetched.final_url):
            external_refs += 1
        else:
            self_refs += 1

    external_form = any(
        action.strip() and not _is_empty_reference(action) and _is_external(action, fetched.final_url)
        for action in parser.form_actions
    )
    self_redirects = 0
    for content in parser.meta_refreshes:
        match = re.search(r"url\s*=\s*([^;]+)$", content, re.I)
        if match and not _is_external(match.group(1).strip(" '\""), fetched.final_url):
            self_redirects += 1

    lines = html.splitlines() or [html]
    searchable = _normal_text(html)
    words = set(searchable.split())
    values: dict[str, object] = {
        "URLLength": url_length,
        "DomainLength": len(hostname),
        "IsDomainIP": _is_ip(hostname),
        "TLD": tld,
        "TLDLength": len(tld),
        "NoOfSubDomain": max(0, len(hostname.split(".")) - 2),
        "HasObfuscation": int(bool(percent_sequences)),
        "NoOfObfuscatedChar": len(percent_sequences),
        "ObfuscationRatio": round(len(percent_sequences) / max(url_length, 1), 6),
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": round(letters / max(url_length, 1), 6),
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": round(digits / max(url_length, 1), 6),
        "NoOfEqualsInURL": fetched.final_url.count("="),
        "NoOfQMarkInURL": fetched.final_url.count("?"),
        "NoOfAmpersandInURL": fetched.final_url.count("&"),
        "NoOfOtherSpecialCharsInURL": specials,
        "SpacialCharRatioInURL": round(specials / max(url_length, 1), 6),
        "IsHTTPS": int(parsed.scheme == "https"),
        "LineOfCode": len(lines),
        "LargestLineLength": max((len(line) for line in lines), default=0),
        "HasTitle": int(bool(title)),
        "DomainTitleMatchScore": domain_title_score,
        "URLTitleMatchScore": url_title_score,
        "HasFavicon": int(parser.has_favicon),
        "Robots": int(parser.has_robots),
        "IsResponsive": int(parser.is_responsive),
        "NoOfURLRedirect": fetched.redirects,
        "NoOfSelfRedirect": self_redirects,
        "HasDescription": int(parser.has_description),
        "NoOfPopup": len(re.findall(r"\b(?:window\.)?open\s*\(", html, re.I)),
        "NoOfiFrame": parser.iframe_count,
        "HasExternalFormSubmit": int(external_form),
        "HasSocialNet": int(social),
        "HasSubmitButton": int(parser.has_submit),
        "HasHiddenFields": int(parser.has_hidden),
        "HasPasswordField": int(parser.has_password),
        "Bank": int(bool(words & BANK_TERMS)),
        "Pay": int(bool(words & PAY_TERMS)),
        "Crypto": int(bool(words & CRYPTO_TERMS)),
        "HasCopyrightInfo": int("copyright" in searchable or "©" in html),
        "NoOfImage": parser.image_count,
        "NoOfCSS": parser.css_count,
        "NoOfJS": parser.js_count,
        "NoOfSelfRef": self_refs,
        "NoOfEmptyRef": empty_refs,
        "NoOfExternalRef": external_refs,
    }
    if any(isinstance(value, float) and not math.isfinite(value) for value in values.values()):
        raise ValueError("Feature extraction produced a non-finite value.")
    audited = pd.DataFrame([[values[name] for name in AUDITED_FEATURE_NAMES]], columns=AUDITED_FEATURE_NAMES)
    return FeatureExtractionResult(
        audited=audited,
        coverage=1.0,
        approximate_features=tuple([
            "DomainTitleMatchScore", "URLTitleMatchScore", "IsResponsive",
            "NoOfSelfRedirect", "Bank", "Pay", "Crypto",
        ]),
        final_url_changed=fetched.final_url != submitted_url,
    )
