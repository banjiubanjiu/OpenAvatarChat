# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OpenAvatarChat** is a modular interactive digital human conversation system that enables real-time voice-based interactions with AI-powered avatars. It supports multiple deployment configurations from local GPU-intensive setups to cloud API-based solutions.

## Development Commands

### Environment Setup
```bash
# Clone with submodules and install git lfs
git clone --recurse-submodules https://github.com/HumanAIGC-Engineering/OpenAvatarChat
git lfs install

# Create virtual environment with uv
uv venv --python 3.11.11

# Install dependencies (all packages)
uv sync --all-packages

# OR install only required dependencies for specific config
uv run install.py --uv --config <config_file_path>.yaml
./scripts/post_config_install.sh --config <config_file_path>.yaml

# Download required models
bash scripts/download_liteavatar_weights.sh
bash scripts/download_musetalk_weights.sh
```

### Running the Application
```bash
# Run with specific configuration
uv run src/demo.py --config <config_file_absolute_path>.yaml

# Common configurations:
uv run src/demo.py --config config/chat_with_lam.yaml
uv run src/demo.py --config config/chat_with_openai_compatible_bailian_cosyvoice.yaml
uv run src/demo.py --config config/chat_with_minicpm.yaml
```

### Docker Commands
```bash
# Build and run with specific config
./build_and_run.sh --config <config_file_relative_path>.yaml

# For CUDA 12.8 (50-series GPUs)
bash build_cuda128.sh
bash run_docker_cuda128.sh --config config/chat_with_openai_compatible_bailian_cosyvoice_musetalk.yaml

# Docker compose
docker compose up    # Start services
docker compose down  # Stop services
```

### Model Management
```bash
# Download LiteAvatar models
bash scripts/download_liteavatar_weights.sh

# Download MuseTalk models
bash scripts/download_musetalk_weights.sh

# Download MiniCPM models
scripts/download_MiniCPM-o_2.6.sh
scripts/download_MiniCPM-o_2.6-int4.sh

# Download avatar models for MuseTalk
python scripts/download_avatar_model.py -m "model_id"
python scripts/download_avatar_model.py -d  # List downloaded models
```

### SSL and Network Setup
```bash
# Generate self-signed SSL certificates
scripts/create_ssl_certs.sh

# Setup coturn TURN server
chmod 777 scripts/setup_coturn.sh
scripts/setup_coturn.sh
```

## Architecture Overview

### Core System Components

**ChatEngine** (`src/chat_engine/`): Central orchestrator that manages all components through HandlerManager, creates sessions, and coordinates data flow between handlers via queues.

**Handler Architecture** (`src/handlers/`): Modular plugin system organized by functional categories:
- **Client Handlers**: WebRTC communication (rtc_client), LAM browser rendering (h5_rendering_client)
- **AI Processing**: ASR (SenseVoice), VAD (SileroVAD), LLMs (MiniCPM, OpenAI-compatible, Qwen-Omni, Dify), TTS (CosyVoice, Edge TTS, Bailian)
- **Avatar Handlers**: LiteAvatar (2D), LAM (3D), MuseTalk (real-time lip-sync)

### Data Flow
```
User Audio → VAD → ASR → LLM → TTS → Avatar → WebRTC → User
```

### Configuration System
YAML-based configuration in `/config/` directory enables flexible deployment:
- Each config specifies which handlers to enable
- Supports local GPU-heavy to lightweight API-based setups
- Environment variables can override sensitive config values via `.env` file

## Key Development Patterns

### Handler Implementation
Each handler implements standardized interfaces:
```python
class HandlerBase(ABC):
    @abstractmethod
    def initialize(self, config: HandlerConfig):
        pass

    @abstractmethod
    def process(self, data: ChatData) -> ChatData:
        pass
```

### Adding New Handlers
1. Create handler directory under `/src/handlers/[category]/[handler_name]/`
2. Implement handler base class interfaces
3. Add `pyproject.toml` for dependencies
4. Create configuration schema
5. Register handler in config file

## Development Notes

### Environment Requirements
- **Python**: 3.11.7+, <3.12 (strict version requirement)
- **CUDA**: 12.4+ for GPU acceleration
- **GPU Memory**: 20GB+ for unquantized MiniCPM, 10GB+ for int4量化
- **Package Management**: UV workspaces for modular dependencies

### Common Issues and Solutions

**CosyVoice on Windows**: Requires Conda environment due to pynini compilation issues
```bash
conda create -n openavatarchat python=3.10
conda activate openavatarchat
conda install -c conda-forge pynini==2.1.6
# Set VIRTUAL_ENV environment variable to Conda path
uv sync --active --all-packages
```

**MuseTalk MMCV Issues**:
```bash
uv pip uninstall mmcv
uv run mim install mmcv==2.2.0 --force
```

**S3FD Model Link for MuseTalk (local)**:
```bash
ln -s $(pwd)/models/musetalk/s3fd-619a316812/* ~/.cache/torch/hub/checkpoints/
```

### Network Requirements
- **SSL Certificates**: Required for non-localhost access due to WebRTC security
- **TURN Server**: Essential for NAT traversal and cross-network connectivity
- **Port Configuration**: Proper firewall configuration for WebRTC, TURN server ports

### Model Storage
Models are stored in `/models/` directory by default. Configuration paths can be absolute or relative to project root.

## Testing and Deployment

### Configuration Testing
Test different configurations based on available resources:
- **Local GPU Mode**: `chat_with_minicpm.yaml` (requires 20GB+ VRAM)
- **API-Hybrid Mode**: `chat_with_openai_compatible_*.yaml` (balanced requirements)
- **Lightweight Mode**: `chat_with_lam.yaml` (minimal GPU, supports multi-session)
- **Multimodal Mode**: `chat_with_qwen_omni.yaml` (end-to-end audio processing)

### Multi-Session Support
LiteAvatar supports concurrent sessions by setting `concurrent_limit` in config. Each session requires ~3GB VRAM when using GPU acceleration.

### Performance Optimization
- Monitor for `[IDLE_FRAME]` warnings indicating frame rate issues
- Adjust `batch_size` for MuseTalk (minimum 2, higher values improve efficiency but increase latency)
- Use `enable_fast_mode` for LiteAvatar to reduce latency at potential cost of smoothness