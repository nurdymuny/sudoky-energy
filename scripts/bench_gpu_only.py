"""GPU-only benchmark: Davis v1 vs v2 vs v3 on the world's hardest puzzles."""
import subprocess, re, os

GPU_DIR = os.path.join(os.path.dirname(__file__), '..', 'sudoku', 'solvers', 'davis_gpu_solver')
EXES = {
    'v1': os.path.join(GPU_DIR, 'davis_solver.exe'),
    'v2': os.path.join(GPU_DIR, 'davis_solver_v2.exe'),
    'v3': os.path.join(GPU_DIR, 'davis_solver_v3.exe'),
}

PUZZLES = {
    'AI Escargot':       '100007090030020008009600500005300900010080002600004000300000010040000007007000300',
    'Inkala 2010':       '800000000003600000070090200050007000000045700000100030001000068008500010090000400',
    'Golden Nugget':     '000000039000001005003050800008090006070002000100400000009080050020000600400700000',
    'Easter Monster':    '100000002090400050006000700050903000000070000000850040700000600030009080002000001',
    'Platinum Blonde':   '000000012000000003002300400001800005060070800000009000008500000900040500470006000',
    'Tarek071':          '000000000000003085001020000000507000004000100090000000500000073002010000000040009',
    '17-clue Coloin':    '000000010400000000020000000000050407008000300001090000300400200050100000000806000',
    'Escargot variant':  '000200000060010700008000050500004000700000002009300000000700400040060009001000030',
    'Norvig hard1':      '400000805030000000000700000020000060000080400000010000000603070500200000104000000',
    'Inkala (Norvig)':   '850002400720000009004000000000107002305000900040000000000080070017000000000036040',
    'champagne 2010':    '000000000000000000009010800000700360060300008001006000080020900000005020070000004',
}


def run_gpu(exe, puzzle):
    try:
        r = subprocess.run([exe, puzzle], capture_output=True, text=True, timeout=30)
        o = r.stdout
        gpu = re.search(r'GPU time:\s+([\d.]+)\s*ms', o)
        sol = re.search(r'Solved:\s*(\d+)\s*/\s*(\d+)', o)
        ph  = re.search(r'P1:\s*(\d+),\s*P2:\s*(\d+),\s*P3:\s*(\d+)', o)
        solved = sol and sol.group(1) == sol.group(2) and int(sol.group(1)) > 0
        ms = float(gpu.group(1)) if gpu else -1
        phase = 'P1' if ph and int(ph.group(1)) > 0 else 'P2' if ph and int(ph.group(2)) > 0 else 'P3'
        return solved, ms, phase
    except Exception as e:
        return False, -1, '?'


def fmt(ok, ms, ph):
    if ok:
        return f"{ms:7.1f} ({ph})"
    return "  FAILED  "


def speedup(ok_a, ms_a, ok_b, ms_b):
    if ok_a and ok_b and ms_b > 0:
        return f"{ms_a / ms_b:.2f}x"
    return "   -  "


def main():
    print("=" * 110)
    print("  Davis GPU Solver Benchmark  —  v1 vs v2 [E1-E6] vs v3 [E1-E8]")
    print("=" * 110)
    print()

    hdr = f"{'Puzzle':<22s}  {'v1 (ms)':>12s}  {'v2 (ms)':>12s}  {'v3 (ms)':>12s}  {'v1->v2':>8s}  {'v2->v3':>8s}  {'v1->v3':>8s}"
    print(hdr)
    print("-" * 110)

    all_results = {}
    for name, puz in PUZZLES.items():
        row = {}
        for ver, exe in EXES.items():
            ok, ms, ph = run_gpu(exe, puz)
            row[ver] = (ok, ms, ph)

        v1ok, v1ms, v1ph = row['v1']
        v2ok, v2ms, v2ph = row['v2']
        v3ok, v3ms, v3ph = row['v3']

        sp12 = speedup(v1ok, v1ms, v2ok, v2ms)
        sp23 = speedup(v2ok, v2ms, v3ok, v3ms)
        sp13 = speedup(v1ok, v1ms, v3ok, v3ms)

        line = (f"{name:<22s}  {fmt(v1ok, v1ms, v1ph):>12s}  "
                f"{fmt(v2ok, v2ms, v2ph):>12s}  {fmt(v3ok, v3ms, v3ph):>12s}  "
                f"{sp12:>8s}  {sp23:>8s}  {sp13:>8s}")
        print(line)
        all_results[name] = row

    print("-" * 110)

    # Averages
    for ver in EXES:
        times = [all_results[n][ver][1] for n in PUZZLES if all_results[n][ver][0]]
        solved = sum(1 for n in PUZZLES if all_results[n][ver][0])
        avg = sum(times) / len(times) if times else 0
        print(f"  {ver}: {solved}/{len(PUZZLES)} solved, avg {avg:.1f} ms")

    print()


if __name__ == "__main__":
    main()
