"""Static audit of ui.html: element ids the JS touches, i18n hygiene, route parity.

Run:  python tools/check_ui.py
Exit: 0 when everything lines up, 1 on any mismatch.
"""
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
html = (ROOT / "ui.html").read_text(encoding="utf-8")
server_src = (ROOT / "server.py").read_text(encoding="utf-8")

problems = []

# --- 1. every $("id") the script references must exist in the markup ----------
markup_end = html.index("<script>")
markup, script = html[:markup_end], html[markup_end:]
ids = set(re.findall(r'\bid="([^"]+)"', markup))
used_ids = set(re.findall(r'\$\("([^"]+)"\)', script))
missing_ids = sorted(used_ids - ids)
if missing_ids:
    problems.append(f'$("...") with no element in markup: {missing_ids}')
print(f"ids in markup {len(ids)} | referenced by script {len(used_ids)} | missing {len(missing_ids)}")

# --- 2. duplicated ids ------------------------------------------------------
dup_ids = sorted(k for k, n in Counter(re.findall(r'\bid="([^"]+)"', html)).items() if n > 1)
if dup_ids:
    problems.append(f"duplicate ids: {dup_ids}")
print("duplicate ids:", dup_ids or "none")

# --- 3. i18n dicts: symmetric, no duplicates, every CJK text node covered ---
pair = re.compile(r'"((?:[^"\\]|\\.)*)":\s*"')


def dicts(text):
    at = text.index("const I18N")
    blk = text[at:]
    h = blk.index('"zh-Hans": {')
    e = blk.index('"en": {')
    return blk[h:e], blk[e:blk.index("\n};", e)]


hans_block, en_block = dicts(html)
hans, eng = pair.findall(hans_block), pair.findall(en_block)
print(f"i18n keys: zh-Hans {len(hans)} | en {len(eng)}")
if len(set(hans)) != len(hans) or len(set(eng)) != len(eng):
    problems.append("duplicate i18n keys")
if set(hans) != set(eng):
    problems.append(f"key sets differ: {sorted(set(hans) ^ set(eng))[:8]}")

cjk = re.compile(r"[一-鿿]")
cands = [m.group(1).strip() for m in re.finditer(r">([^<>{}]+)<", markup)]
cands += re.findall(r'placeholder="([^"]*)"', markup)
untranslated = []
for text in dict.fromkeys(cands):
    if text and cjk.search(text) and text not in set(hans):
        untranslated.append(text)
print("markup strings without translation:", len(untranslated))
for text in untranslated:
    print("   ", text)

# --- 4. /api/image/* calls must be served by server.py ----------------------
called = sorted(set(re.findall(r'["\'`](/api/image/[a-z/]+)', script)))
served = set(re.findall(r'"(/api/image/[a-z/]+)"', server_src))
unserved = [p for p in called if p not in served]
if unserved:
    problems.append(f"UI calls routes the server lacks: {unserved}")
print(f"image routes: ui {len(called)} | server {len(served)} | unserved {unserved}")

# --- 5. balanced <div> tags (cheap structural sanity) ----------------------
opens = len(re.findall(r"<div\b", markup))
closes = len(re.findall(r"</div>", markup))
print(f"div tags: {opens} open / {closes} close")
if opens != closes:
    problems.append(f"unbalanced div tags: {opens} vs {closes}")

# --- 6. JS parses under node, when node is available ----------------------
try:
    tmp = ROOT / "_check_ui_script.js"
    tmp.write_text(script[script.index(">") + 1:script.rindex("</script>")], encoding="utf-8")
    done = subprocess.run(["node", "--check", str(tmp)], capture_output=True)
    tmp.unlink(missing_ok=True)
    print("node --check:", "ok" if done.returncode == 0 else done.stderr.decode()[:400])
    if done.returncode != 0:
        problems.append("ui.html script does not parse")
except FileNotFoundError:
    print("node --check: skipped (node not installed)")

print()
if problems:
    for item in problems:
        print("[fail]", item)
    sys.exit(1)
print("[ok] ui.html audit passed")
