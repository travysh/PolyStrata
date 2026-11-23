# pplog.py SERVER_PY OUT: log every PP (prompt progress) line the server gets from the engine, with the time, to OUT
import re, sys
p, out = sys.argv[1], sys.argv[2]
s = open(p).read()
s2, n = re.subn(r"^(\s*)(elif|if) line\.startswith\(\"PP \"\):\n",
                lambda m: m.group(0) + m.group(1) + "    open(%r, \"a\").write(\"%%.3f %%s\\n\" %% (__import__(\"time\").time(), line.rstrip()))\n" % out,
                s, flags=re.M)
open(p, "w").write(s2)
print("patched", n)
