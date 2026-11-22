# Session 1 — LLM 추론의 기초 & 첫 SGLang 서버

> **핵심 질문:** token 하나는 어떻게 만들어지고, 왜 `model.generate()`만으로는 서빙을 못 하는가?

## 학습 목표

이 회차가 끝나면 다음을 **자기 말로, 숫자를 섞어서** 설명할 수 있다.

- [ ] autoregressive loop 한 바퀴 (tokenize → forward → logits → sample → append)
- [ ] prefill과 decode가 왜 다른 종류의 일인지 (compute-bound vs memory-bound)
- [ ] KV cache가 없으면 무엇이 반복 계산되는지, `config.json`으로 KV 크기를 계산하는 법
- [ ] "batch=1 decode의 이론적 최저 시간 ≈ weight bytes / 메모리 대역폭"
- [ ] TTFT / ITL / TPOT / E2E / throughput 정의와 측정 위치
- [ ] SGLang 서버를 띄우고 native API와 OpenAI 호환 API로 요청을 보내는 법

## 타임테이블 (160분)

| 시간 | 순서 | 자료 |
|---|---|---|
| 0:00–0:25 | **오리엔테이션** (발표 대신): 스터디 목표·원칙·진행 방식·예산 규칙, RunPod 계정/크레딧 확인 | [../README.md](../README.md), [../docs/runpod_guide.md](../docs/runpod_guide.md) |
| 0:25–1:15 | **이론** §1–§7 | [lecture.md](lecture.md) |
| 1:15–1:25 | 휴식 — **이때 Pod Deploy + `setup_pod.sh` 시작** | [lab.md §0](lab.md#0-pod-준비-휴식-시간에) |
| 1:25–2:30 | **실습** 1–5 | [lab.md](lab.md) |
| 2:30–2:40 | **과제 안내** | [assignment.md](assignment.md) |

## 사전 준비 (수강생, 수업 전 집에서)

- [ ] RunPod 가입, 크레딧 $20 확인
- [ ] 이 레포 fork → 로컬 clone
- [ ] 로컬에서 `python common/mock_server.py` + `python common/client.py --input-len 128` 이 동작하는지 확인
      (Python 3.10+, `pip install requests matplotlib pandas`)
- [ ] (선택) "Attention Is All You Need"의 Figure 1, 또는 Transformer decoder 구조 그림 한 번 보기

## 강사 체크리스트

- [ ] 수업 전날 [lab.md](lab.md)를 직접 끝까지 리허설하고 **본인 숫자로 예시 표 채워 두기** (lecture의 "예측 질문" 정답 확인용)
- [ ] CUDA 13 필터·템플릿 방식(A/B) 확정 → 수강생 공지
- [ ] 이론 중 §4(roofline)는 화이트보드로 직접 계산하며 진행 — 이 회차의 핵심
- [ ] 실습 종료 10분 전 "결과 다운로드 → Terminate" 일괄 안내

## 이번 회차 GPU 예산: $1.8 (수업 $0.9 + 과제 $0.9)
