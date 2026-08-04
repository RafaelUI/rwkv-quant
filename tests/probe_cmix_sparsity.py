"""
Разреженность скрытого слоя channel-mix: сколько трафика cmix.value можно
не читать при декоде.

Почему это вообще возможно: RWKV-7 считает k = relu(x @ key.T)**2, то есть
ТОЧНЫЕ нули там, где до-активация отрицательна. Дальше идёт k @ value.T --
и колонки `value`, попавшие на нулевые k, на результат не влияют вовсе.
cmix.value -- это ~23-28% модели, так что потенциал заметный.

Но GPU читает не колонками, а блоками: в нашем sb6 блок из 32 колонок =
26 байт одним лоадом. При РАВНОМЕРНОЙ разреженности 50% в каждом блоке
из 32 найдётся ~16 ненулевых, и пропустить нельзя ни одного блока --
выигрыш ноль. Поэтому меряется не только доля нулей, но и:

  - доля блоков из 32, целиком нулевых (что можно пропустить точно);
  - концентрация энергии: какая доля ||k||^2 лежит в top-N% блоков
    (что можно пропустить приближённо, если хвост несёт мало).

Первая цифра -- потолок точного пропуска, вторая -- потолок
приближённого (top-k блоков вместо всех).

Запуск: python tests/probe_cmix_sparsity.py [n_seq]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from rwkv_quant.models.rwkv7_ref import RWKV7Ref  # noqa: E402
import rwkv_quant.models.rwkv7_ref as ref  # noqa: E402

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
GS = 32          # ширина блока в формате sb6
N_SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 2

STATS = {"zeros": 0.0, "elems": 0, "zero_blocks": 0, "blocks": 0,
         "zero_blocks_perm": 0,
         "energy_top": {q: 0.0 for q in (0.125, 0.25, 0.5)}, "n_tok": 0}


def patched_cmix(self, x, c, cfg, layer_id=-1):
    xx = self._time_shift(x) - x
    kx = x + xx * c.x_k
    k = torch.relu(kx @ ref.q(c.key, "cmix", cfg).T) ** 2   # [B, T, H]

    f = k.reshape(-1, k.shape[-1]).float()                   # [BT, H]
    STATS["zeros"] += (f == 0).sum().item()
    STATS["elems"] += f.numel()
    STATS["n_tok"] += f.shape[0]

    nb = f.shape[1] // GS
    blk = f[:, :nb * GS].reshape(f.shape[0], nb, GS)
    STATS["zero_blocks"] += (blk.abs().amax(dim=2) == 0).sum().item()
    STATS["blocks"] += f.shape[0] * nb

    # энергия по блокам: сколько ||k||^2 несут самые сильные блоки
    e = (blk ** 2).sum(dim=2)                                # [BT, nb]
    tot = e.sum(dim=1).clamp_min(1e-30)
    srt = torch.sort(e, dim=1, descending=True).values
    for qf in STATS["energy_top"]:
        top = srt[:, :max(1, int(nb * qf))].sum(dim=1)
        STATS["energy_top"][qf] += (top / tot).sum().item()

    # Что даст ПЕРЕСТАНОВКА каналов. Перестановка колонок cmix.value и
    # строк cmix.key -- офлайн и математически тождественна модели, но
    # меняет то, какие каналы попадают в один блок из 32. Сортируем по
    # частоте активации: редко активные каналы съезжают в одни блоки и
    # эти блоки чаще оказываются целиком нулевыми.
    # Оценка оптимистична: перестановка выведена на тех же токенах.
    nzm = (f != 0)                                            # [BT, H]
    freq = nzm.float().mean(dim=0)                            # [H]
    perm = torch.argsort(freq, descending=True)
    fp = nzm[:, perm][:, :nb * GS].reshape(f.shape[0], nb, GS)
    STATS["zero_blocks_perm"] += (~fp.any(dim=2)).sum().item()

    return k @ ref.q(c.value, "cmix", cfg).T


def main():
    RWKV7Ref._cmix_forward = patched_cmix
    model = RWKV7Ref(CKPT, device="cpu", dtype=torch.bfloat16)
    data = torch.load(CORPUS)["tokens"][:N_SEQ].long()
    with torch.no_grad():
        for i in range(data.shape[0]):
            model.forward(data[i:i + 1, :-1])
            print(f"  seq {i+1}/{data.shape[0]}", flush=True)

    z = STATS["zeros"] / STATS["elems"]
    zb = STATS["zero_blocks"] / STATS["blocks"]
    print(f"\nтокенов x слоёв: {STATS['n_tok']}")
    print(f"нулей в k (relu^2):            {100*z:6.2f}%")
    zbp = STATS["zero_blocks_perm"] / STATS["blocks"]
    print(f"блоков по {GS} целиком нулевых: {100*zb:6.2f}%  "
          f"<- столько cmix.value можно НЕ читать точно")
    print(f"  то же после перестановки:    {100*zbp:6.2f}%  "
          f"<- каналы отсортированы по частоте активации")
    print(f"  для справки, будь каналы независимы: "
          f"{100*(z**GS):6.2f}%")
    print("\nконцентрация энергии по блокам (приближённый пропуск):")
    for qf in sorted(STATS["energy_top"]):
        share = STATS["energy_top"][qf] / STATS["n_tok"]
        print(f"  top-{100*qf:4.1f}% блоков несут {100*share:6.2f}% ||k||^2")


if __name__ == "__main__":
    main()
