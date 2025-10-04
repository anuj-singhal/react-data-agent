# MCP Oracle Client

A modular, scalable Oracle database client using Model Context Protocol (MCP) with natural language to SQL conversion.

## Project Structure

```
mcp_oracle_client/
├── main.py                 # Application entry point
├── api/                    # FastAPI application
│   ├── app.py             # App initialization
│   ├── middleware.py      # Middleware configuration
│   └── lifespan.py        # Startup/shutdown management
├── config/                # Configuration
│   ├── settings.py        # Settings management
│   └── logging_config.py  # Logging setup
├── routers/               # API endpoints
│   ├── health.py          # Health check routes
│   ├── query.py           # Query execution routes
│   └── database.py        # Database info routes
├── services/              # Business logic
│   ├── mcp_client.py      # MCP client manager
│   ├── sql_converter.py   # NL to SQL conversion
│   └── query_executor.py  # Query execution
├── models/                # Data models
│   ├── requests.py        # Request models
│   └── responses.py       # Response models
├── data/                  # Data definitions
│   └── schemas.py         # Database schemas
└── utils/                 # Utilities
    ├── constants.py       # Constants
    ├── helpers.py         # Helper functions
    └── validators.py      # Input validators
```

## Installation

1. **Clone the repository**
```bash
git clone <repository>
cd mcp_oracle_client
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

## Configuration

Edit the `.env` file with your settings:

- `MCP_SERVER_PATH`: Path to your Oracle MCP server script
- `OPENAI_API_KEY`: (Optional) OpenAI API key for better SQL conversion
- `PORT`: Server port (default: 8000)

## Usage

### Start the server
```bash
python main.py
```

### Access the API
- API: `http://localhost:8000`
- Documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### API Endpoints

#### Execute Query
```bash
POST /query
{
  "query": "Show all tables"
}
```

#### Get Tables
```bash
GET /tables
```

#### Get Table Info
```bash
GET /table/{table_name}
```

#### Health Check
```bash
GET /health
```

## Development

### Adding New Features

1. **New Endpoint**: Add router in `routers/`
2. **New Service**: Add service in `services/`
3. **New Model**: Add model in `models/`
4. **Configuration**: Update `config/settings.py`

### Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=.
```

### Code Style

```bash
# Format code
black .

# Lint
flake8
```

## Architecture

### Separation of Concerns

- **Routers**: Handle HTTP requests/responses
- **Services**: Business logic and external integrations
- **Models**: Data validation and serialization
- **Config**: Centralized configuration
- **Utils**: Shared utilities

### Benefits

- **Modular**: Each component has single responsibility
- **Scalable**: Easy to add new features
- **Maintainable**: Clear structure and separation
- **Testable**: Components can be tested independently
- **Configurable**: Environment-based configuration

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### Production

For production deployment:
1. Set `RELOAD=false` in `.env`
2. Use proper logging configuration
3. Set up monitoring
4. Configure proper database credentials

## License

MIT