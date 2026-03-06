# GenesisOSCBridge 플러그인

**Genesis OSC Bridge**는 언리얼 엔진 5(Unreal Engine 5)와 파이썬 기반의 물리 시뮬레이터인 **Genesis** 간의 양방향 실시간 물리 환경 동기화를 지원하는 통신(OSC) 플러그인입니다. 

같이 제공되는 `kick_ball.py` 및 `osc_manager.py` 는 이 플러그인이 어떻게 작동하는지(초기화 대기, 지형 생성, 실시간 위치 연동) 직관적으로 보여주기 위한 데모 파이썬 스크립트입니다.

---

## 시작 가이드

### 0. 필수 라이브러리 설치
파이썬 스크립트 실행에 앞서 OSC 통신을 위한 파이썬 라이브러리를 반드시 설치해야 합니다. 터미널(명령 프롬프트)을 열고 다음 명령어를 입력하세요.
```bash
pip install python-osc
```

### 1. 언리얼 엔진 환경 세팅
1. **플러그인 활성화**: 언리얼 프로젝트에 플러그인 소스를 복사/적용합니다.
2. **액터 배치**: 레벨(맵)에 `GenesisOSCBridge` 액터를 드래그하여 배치합니다.
3. **타겟 연결**: 시뮬레이터의 움직임(예: 공)을 받을 대상 액터를 `TargetActor` 변수에 할당합니다.
4. **장애물 태그 설정**: 제네시스 환경과 동기화하여 똑같은 물리 구조물(충돌체)로 만들고 싶은 액터들(StaticMesh, Box 등)을 선택한 뒤, 디테일 패널의 **Tags(태그)** 항목에 `GenesisObstacle` 이라는 태그를 반드시 추가합니다.

### 2. 실행 순서
1. 터미널에서 `python kick_ball.py`를 실행하여 제네시스를 대기 상태로 만듭니다.
2. 언리얼 엔진에서 **Play**를 클릭합니다.
3. 언리얼이 수집한 'GenesisObstacle' 구조물 데이터를 제네시스로 전송하고, 제네시스는 해당 구조물들을 생성한 뒤 공을 떨어뜨리며 60Hz 단위로 언리얼 엔진에 실시간 렌더링을 지시합니다.

---

## 플러그인 핵심 구현 함수 설명

이 플러그인(`GenesisOSCBridge.cpp/h`)은 블루프린트 및 C++에서 호출할 수 있는 기능들을 제공합니다. 주요 함수들은 다음과 같습니다.

### 1. `InitializeConnection(TargetIP, SendPort, RecvPort, RecvPortObs)`
양방향 OSC 통신을 시작하는 핵심 초기화 함수입니다.
* **가동 시점**: `BeginPlay()` 내부에서 자동 호출됩니다.
* **기능**: 패킷을 보낼 대상 서버(`GenesisClient`)와 패킷을 받을 수신 서버(`GenesisServer`)를 지정된 IP와 포트로 바인딩(Bind)하여 엽니다. 기존 연결이 있다면 안전하게 닫고 재연결합니다.

### 2. `SendInitPhysics(FGenesisOSCPhysicsSettings Settings)`
제네시스 시뮬레이터 쪽에 물리 환경 기초값을 세팅하라고 명령하는 함수입니다.
* **기능**: 중력(Gravity), 시간 간격(TimeStep), 마찰 계수(Friction) 등을 OSC 패킷(`/Genesis/Init/Physics`)으로 묶어 파이썬으로 보냅니다. 파이썬에서는 이 값을 받아 시뮬레이터 환경 구성에 참고할 수 있습니다.

### 3. `GatherLevelObstacles()`
언리얼 엔진의 환경(맵)을 제네시스 가상 환경에 똑같이 복제하기 위해 호출되는 씬(Scene) 동기화 함수입니다.
* **기능**: 현재 레벨 내에서 `GenesisObstacle` 태그를 가진 모든 액터(AActor)를 추려냅니다.
* **최적화 로직**: 
  * 물체가 회전하더라도 엉뚱한 크기로 왜곡(World AABB 이슈)되지 않도록, **`SMC->GetBoundingBox().GetSize()`를 통해 고유의 로컬 크기(Local Bounds)를 추출**하고 Actor Scale을 곱해 정확한 물리 크기를 산출합니다.
  * 구(Sphere), 실린더(Cylinder), 박스(Box) 등 액터의 컴포넌트나 이름 특성을 파악하여 고유한 `Type` ID값과 함께 위치, 회전값(Quat), 크기(Dimensions)를 `/Init/Obstacle` 주소로 전송합니다.
  * 현재 버전은 기본적인 박스, 실린더, 구, 평면만 지원하며 복잡한 구조물은 지원하지 않습니다.

### 4. `HandleOSCMessage(const FOSCMessage& Message)`
파이썬(제네시스) 측에서 실시간 연산 결과를 수신받는 콜백(Callback) 이벤트 리스너입니다.
* **Location / Rotation**: `TargetActor`의 월드 좌표와 쿼터니언 회전값을 받아옵니다.
* **Tick() 동기화**: 수신받은 좌표가 있으면 `Tick(DeltaTime)` 단계에서 `TargetActor->SetActorLocationAndRotation(ETeleportType::TeleportPhysics)`을 호출해 잔상(Ghosting) 없이 캐릭터/메쉬 데이터를 부드럽게 실시간 텔레포트(이동) 지킵니다.

### 5. `SendCommand(EGenesisOSCCommand Command)`
시뮬레이션의 흐름을 원격으로 제어하는 리모컨 역할을 합니다.
* **명령어 종류**: `Start` (시작), `Pause` (일시 정지), `Stop` (종료), `Reset` (초기화)
* **기능**: 언리얼 엔진 안에서 블루프린트나 C++ 코드로 이 함수를 호출하여 상태를 제어합니다. 예를 들어 `Reset` 커맨드를 전송하면, 사용자의 정의에 따른 초기 설정값으로 복구됩니다.
