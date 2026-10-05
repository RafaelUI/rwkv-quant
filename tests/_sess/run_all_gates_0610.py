"""06.10: все гейты tests/test_*.py подряд, без аргументов, с пределом времени на гейт; коды возврата в журнал.
Ожидание (шапка NEXT_SESSION 03.10): 0 -- зелёный, 2 -- справочный без RWKVQ_GATE_QKL; три гейта требуют аргументов
(k3_from_canonical, wkv_var_model, measure_multidev) и здесь честно падают / пропускаются.
    python tests/_sess/run_all_gates_0610.py <журнал> [предел, с]"""
import glob, os, subprocess, sys, time
LOG = sys.argv[1]; LIM = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tests = sorted(glob.glob(os.path.join(root, "tests", "test_*.py")))
os.makedirs(LOG + ".d", exist_ok=True)
with open(LOG, "a") as f: f.write("старт %s, гейтов %d, RWKVQ_RKV_SHARE=%s\n" % (time.ctime(), len(tests), os.environ.get("RWKVQ_RKV_SHARE", "(умолчание)")))
for t in tests:
    n = os.path.basename(t)[:-3]; t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-u", t], cwd=root, stdout=open(os.path.join(LOG + ".d", n + ".log"), "w"), stderr=subprocess.STDOUT, timeout=LIM)
        code = str(r.returncode)
    except subprocess.TimeoutExpired:
        code = "ПРЕДЕЛ"
    with open(LOG, "a") as f: f.write("%-44s код %-7s %5.0f с\n" % (n, code, time.time() - t0))
with open(LOG, "a") as f: f.write("конец %s\n" % time.ctime())
