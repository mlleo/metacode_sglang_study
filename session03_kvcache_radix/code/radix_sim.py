"""SGLang `RadixCache`(mem_cache/radix_cache.py)의 핵심 알고리즘을 80줄로 줄인 교육용 시뮬레이터. GPU 불필요.

- 노드는 token 시퀀스 조각(key)과 그 token들의 KV 슬롯 번호(value)를 가진다
- match: 루트부터 내려가며 최장 공통 prefix를 찾고, 노드 중간에서 끝나면 split
- insert: 남은 부분을 새 leaf로 붙인다 (중간에서 갈라지면 split)
- evict: lock_ref == 0 인 leaf를 last_access_time이 오래된 순(LRU)으로 지운다
- 요청은 마지막 token을 제외한 부분만 match한다 (SGLang: max_prefix_len = input_len - 1)

과제의 '손 추적' 답을 검증할 때 쓰세요.

    python session03_kvcache_radix/code/radix_sim.py ABCDE ABCXY ABCDEZ ABQR --capacity 14
    python session03_kvcache_radix/code/radix_sim.py "1 2 3 4 5" "1 2 3 8 9" --capacity 100
"""

from __future__ import annotations

import argparse
import itertools

_clock = itertools.count()


class Node:
    def __init__(self, key=(), value=(), parent=None):
        self.key, self.value, self.parent = list(key), list(value), parent
        self.children: dict = {}     # 첫 token → 자식 노드 (SGLang: child_key)
        self.lock_ref = 0
        self.last_access = next(_clock)


class RadixTree:
    def __init__(self, capacity: int):
        self.root = Node()
        self.free = list(range(capacity))   # 빈 KV 슬롯 (SGLang: TokenToKVPoolAllocator)

    # ---- match ---------------------------------------------------------
    def match(self, key):
        node, matched = self.root, []
        node.last_access = next(_clock)
        while key and key[0] in node.children:
            child = node.children[key[0]]
            child.last_access = next(_clock)
            n = _common(child.key, key)
            if n < len(child.key):              # 노드 중간에서 갈라짐 → split
                child = self._split(child, n)
                matched += child.value
                return matched, child
            matched += child.value
            node, key = child, key[n:]
        return matched, node

    def _split(self, child: Node, n: int) -> Node:
        new = Node(child.key[:n], child.value[:n], child.parent)
        new.lock_ref = child.lock_ref
        new.parent.children[new.key[0]] = new
        child.key, child.value, child.parent = child.key[n:], child.value[n:], new
        new.children = {child.key[0]: child}
        return new

    # ---- insert --------------------------------------------------------
    def insert(self, key, value):
        node = self.root
        while key and key[0] in node.children:
            child = node.children[key[0]]
            child.last_access = next(_clock)
            n = _common(child.key, key)
            if n < len(child.key):
                child = self._split(child, n)
            node, key, value = child, key[n:], value[n:]
        if key:
            leaf = Node(key, value, node)
            node.children[key[0]] = leaf
            node = leaf
        return node

    # ---- evict (LRU leaf first) ---------------------------------------
    def evict(self, need: int) -> list:
        evicted = []
        while len(self.free) < need:
            leaves = [n for n in self._nodes() if not n.children and n is not self.root and n.lock_ref == 0]
            if not leaves:
                raise MemoryError("evict할 수 있는 leaf가 없습니다 (전부 lock됨) → SGLang이라면 요청을 대기시키거나 retract")
            victim = min(leaves, key=lambda n: n.last_access)
            self.free += victim.value
            evicted.append("".join(map(str, victim.key)))
            del victim.parent.children[victim.key[0]]
        return evicted

    def alloc(self, n: int):
        evicted = self.evict(n)
        slots, self.free = self.free[:n], self.free[n:]
        return slots, evicted

    def _nodes(self):
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack += n.children.values()

    def dump(self, node=None, depth=0):
        node = node or self.root
        for c in sorted(node.children.values(), key=lambda c: str(c.key[0])):
            k = "".join(map(str, c.key)) if all(len(str(t)) == 1 for t in c.key) else " ".join(map(str, c.key))
            print("    " + "  " * depth + f"└─ [{k}] slots={c.value} lock={c.lock_ref} t={c.last_access}")
            self.dump(c, depth + 1)


def _common(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def parse(s: str):
    return s.split() if " " in s else list(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("requests", nargs="+", help='요청 token 시퀀스. "ABCDE" 또는 "1 2 3 4 5"')
    p.add_argument("--capacity", type=int, default=64, help="KV pool 슬롯 수")
    args = p.parse_args()

    tree = RadixTree(args.capacity)
    for i, s in enumerate(args.requests, 1):
        toks = parse(s)
        matched, last = tree.match(toks[:-1])          # 마지막 token은 항상 다시 계산
        last.lock_ref += 1                              # 실행 중에는 evict 금지 (inc_lock_ref, 단순화)
        new_slots, evicted = tree.alloc(len(toks) - len(matched))
        tree.insert(toks, matched + new_slots)          # 요청 종료 시 전체를 tree에 등록 (cache_finished_req)
        last.lock_ref -= 1
        print(f"req{i} {s!r}: cached={len(matched)} new={len(new_slots)}"
              + (f"  evicted={evicted}" if evicted else "") + f"  free={len(tree.free)}")
        tree.dump()


if __name__ == "__main__":
    main()
