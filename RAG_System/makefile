# Makefile for RAG System Management

.PHONY: help setup install run-local run-docker test clean backup restore monitor

# Default target
help:
	@echo "RAG System Management Commands:"
	@echo "  make setup          - Initial setup (create dirs, copy .env)"
	@echo "  make install        - Install Python dependencies"
	@echo "  make run-local      - Run pipeline locally"
	@echo "  make run-interface  - Start web interface"
	@echo "  make run-docker     - Run complete stack with Docker"
	@echo "  make test           - Run tests"
	@echo "  make clean          - Clean cache and temporary files"
	@echo "  make backup         - Backup database and configurations"
	@echo "  make restore        - Restore from backup"
	@echo "  make monitor        - Open monitoring dashboards"
	@echo "  make status         - Show system status"
	@echo "  make logs           - Show recent logs"

# Setup environment
setup:
	@echo "Setting up RAG system..."
	@mkdir -p data_cache logs backups monitoring/grafana/dashboards init_scripts
	@if [ ! -f .env ]; then cp .env.template .env && echo "Created .env file - please edit with your settings"; fi
	@chmod +x setup.py
	@python setup.py

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Downloading embedding model..."
	python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Run pipeline locally
run-local:
	@echo "Running RAG pipeline locally..."
	python pipeline_orchestrator.py --mode auto

# Run in full mode
run-full:
	@echo "Running full pipeline rebuild..."
	python pipeline_orchestrator.py --mode full --force

# Run incremental update
run-incremental:
	@echo "Running incremental update..."
	python pipeline_orchestrator.py --mode incremental

# Start web interface
run-interface:
	@echo "Starting web interface on http://localhost:7860"
	python 04_chat_interface.py

# Docker operations
run-docker:
	@echo "Starting Docker stack..."
	docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 10
	@echo "Stack is running. Access points:"
	@echo "  - Web Interface: http://localhost:7860"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"
	@echo "  - Prometheus: http://localhost:9090"

stop-docker:
	@echo "Stopping Docker stack..."
	docker-compose down

rebuild-docker:
	@echo "Rebuilding Docker images..."
	docker-compose build --no-cache

# Testing
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

test-connection:
	@echo "Testing database connection..."
	python -c "from config import DB_URI; from sqlalchemy import create_engine; engine = create_engine(DB_URI); engine.connect(); print('✅ Database connection successful')"

# Cleaning
clean:
	@echo "Cleaning temporary files..."
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-cache:
	@echo "Cleaning data cache..."
	rm -rf data_cache/*
	rm -rf logs/*

clean-all: clean clean-cache
	@echo "Full cleanup complete"

# Backup operations
backup:
	@echo "Creating backup..."
	@mkdir -p backups/$(shell date +%Y%m%d)
	@if [ -f .env ]; then cp .env backups/$(shell date +%Y%m%d)/; fi
	@if [ -d data_cache ]; then tar -czf backups/$(shell date +%Y%m%d)/data_cache.tar.gz data_cache/; fi
	@echo "Backing up database..."
	@docker-compose exec -T postgres pg_dump -U raguser ragdb > backups/$(shell date +%Y%m%d)/ragdb.sql
	@echo "Backup created in backups/$(shell date +%Y%m%d)/"

restore:
	@echo "Available backups:"
	@ls -la backups/
	@echo "To restore, run: make restore-date DATE=YYYYMMDD"

restore-date:
	@if [ -z "$(DATE)" ]; then echo "Please specify DATE=YYYYMMDD"; exit 1; fi
	@echo "Restoring from backup $(DATE)..."
	@if [ -f backups/$(DATE)/.env ]; then cp backups/$(DATE)/.env .; fi
	@if [ -f backups/$(DATE)/data_cache.tar.gz ]; then tar -xzf backups/$(DATE)/data_cache.tar.gz; fi
	@if [ -f backups/$(DATE)/ragdb.sql ]; then docker-compose exec -T postgres psql -U raguser ragdb < backups/$(DATE)/ragdb.sql; fi
	@echo "Restore complete"

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:3000 || open http://localhost:3000

metrics:
	@echo "Current metrics:"
	@curl -s http://localhost:8000/metrics | grep "rag_system_" | grep -v "#"

# Status and logs
status:
	@echo "System Status:"
	@echo "=============="
	python pipeline_orchestrator.py --status

logs:
	@echo "Recent logs:"
	@echo "============"
	@tail -n 50 logs/rag_system_$(shell date +%Y%m%d).log

logs-follow:
	@tail -f logs/rag_system_$(shell date +%Y%m%d).log

# Development helpers
format:
	@echo "Formatting code..."
	black *.py

lint:
	@echo "Linting code..."
	flake8 *.py --max-line-length=100

# Database management
db-shell:
	@echo "Connecting to database..."
	@docker-compose exec postgres psql -U raguser ragdb

db-stats:
	@echo "Database statistics:"
	@docker-compose exec postgres psql -U raguser ragdb -c "\
		SELECT \
			COUNT(*) as total_vectors, \
			COUNT(DISTINCT source_table) as tables, \
			COUNT(DISTINCT semantic_topic) as topics, \
			pg_size_pretty(pg_total_relation_size('rag_documents_final')) as table_size \
		FROM rag_documents_final;"

# Performance tuning
optimize-index:
	@echo "Optimizing vector index..."
	@docker-compose exec postgres psql -U raguser ragdb -c "\
		REINDEX INDEX idx_rag_documents_final_embedding; \
		VACUUM ANALYZE rag_documents_final;"

# Quick start for new users
quickstart: setup install
	@echo "QuickStart complete! Next steps:"
	@echo "1. Edit .env with your database credentials"
	@echo "2. Ensure PostgreSQL is running with pgvector extension"
	@echo "3. Run: make run-local"
	@echo "4. In another terminal: make run-interface"