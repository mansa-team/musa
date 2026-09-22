import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_EXE = os.path.normpath(os.path.join(HERE, "llamacpp-bin", "llama-server.exe"))
LOG = os.path.join(HERE, "server.log")
PIDFILE = os.path.join(HERE, "server.pid")

if len(sys.argv) > 1 and sys.argv[1] == "stop":
    with open(PIDFILE) as f:
        pid = f.read().strip()
    subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
    try:
        os.remove(PIDFILE)
    except OSError:
        pass
    print("stopped " + pid)
    sys.exit(0)

flags = sys.argv[1:]
logf = open(LOG, "ab")
p = subprocess.Popen(
    [SERVER_EXE] + flags,
    stdin=subprocess.DEVNULL,
    stdout=logf,
    stderr=subprocess.STDOUT,
    creationflags=0x00000008 | 0x00000200,  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    close_fds=True,
)
with open(PIDFILE, "w") as f:
    f.write(str(p.pid))
print("launched pid=" + str(p.pid))
