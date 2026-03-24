#!/bin/bash

# Antenna ML Docker Helper Script
# Makes common tasks easier to run

case "$1" in
    start)
        echo "🚀 Starting Docker container..."
        sudo docker compose up -d
        echo "✅ Container started!"
        sudo docker compose ps
        ;;
    
    stop)
        echo "🛑 Stopping Docker container..."
        sudo docker compose down
        echo "✅ Container stopped!"
        ;;
    
    restart)
        echo "🔄 Restarting Docker container..."
        sudo docker compose restart
        echo "✅ Container restarted!"
        ;;
    
    rebuild)
        echo "🔨 Rebuilding Docker container..."
        sudo docker compose down
        sudo docker compose build
        sudo docker compose up -d
        echo "✅ Container rebuilt and started!"
        ;;
    
    train)
        echo "🎓 Training antenna model..."
        sudo docker compose exec ml python train_model.py
        ;;
    
    tune)
        echo "🔧 Running hyperparameter tuning..."
        echo "⚠️  This will take 5-10 minutes..."
        sudo docker compose exec ml python tune_hyperparameters.py
        ;;

    test)
        echo "🧪 Running test suite..."
        sudo docker compose exec ml python -m pytest tests/ -v --cov
        ;;
    
    app)
        echo "🌐 Starting Gradio web app..."
        echo "📱 Access at: http://localhost:7860"
        echo "Press Ctrl+C to stop"
        echo ""
        sudo docker compose exec ml python gradio_app.py
        ;;
    
    shell)
        echo "🐚 Opening bash shell in container..."
        sudo docker compose exec ml bash
        ;;
    
    python)
        echo "🐍 Opening Python shell in container..."
        sudo docker compose exec ml python
        ;;
    
    logs)
        echo "📋 Viewing container logs..."
        echo "Press Ctrl+C to stop following"
        sudo docker compose logs -f
        ;;
    
    status)
        echo "📊 Container status:"
        sudo docker compose ps
        echo ""
        echo "📁 Output files:"
        ls -lh *.pkl *.png 2>/dev/null || echo "  No output files yet"
        ;;
    
    clean)
        echo "🧹 Cleaning up output files..."
        read -p "⚠️  This will delete all .pkl and .png files. Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f *.pkl *.png
            echo "✅ Cleaned!"
        else
            echo "❌ Cancelled"
        fi
        ;;
    
    clean-all)
        echo "🧹 Deep cleaning (containers + images + outputs)..."
        read -p "⚠️  This will remove containers, images, and output files. Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo docker compose down --rmi all
            rm -f *.pkl *.png
            echo "✅ Deep clean complete!"
        else
            echo "❌ Cancelled"
        fi
        ;;
    
    exec)
        if [ -z "$2" ]; then
            echo "❌ Usage: ./run.sh exec <command>"
            echo "Example: ./run.sh exec 'python train_model.py'"
        else
            echo "⚡ Executing: $2"
            sudo docker compose exec ml bash -c "$2"
        fi
        ;;
    
    update)
        echo "📦 Updating Python packages..."
        echo "Edit requirements.txt first, then run this"
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ./run.sh rebuild
        else
            echo "❌ Cancelled"
        fi
        ;;
    
    upload)
        if [ -z "$2" ]; then
            echo "📤 Upload a file to container"
            echo "Usage: ./run.sh upload <filename>"
            echo "Example: ./run.sh upload new_dataset.csv"
        else
            if [ -f "$2" ]; then
                echo "📤 Uploading $2 to container..."
                sudo docker cp "$2" antenna_ml:/app/
                echo "✅ Upload complete!"
            else
                echo "❌ File not found: $2"
            fi
        fi
        ;;
    
    download)
        if [ -z "$2" ]; then
            echo "📥 Download a file from container"
            echo "Usage: ./run.sh download <filename>"
            echo "Example: ./run.sh download rf_antenna_model.pkl"
        else
            echo "📥 Downloading $2 from container..."
            sudo docker cp antenna_ml:/app/"$2" .
            echo "✅ Download complete!"
        fi
        ;;
    
    build-exe)
        echo "📦 Building standalone executable with PyInstaller..."
        sudo docker compose exec ml python build_exe.py
        echo "✅ Build complete! Output in dist/antenna-ml/"
        ;;
    
    backup)
        timestamp=$(date +%Y%m%d_%H%M%S)
        backup_dir="backup_${timestamp}"
        echo "💾 Creating backup..."
        mkdir -p "$backup_dir"
        cp *.pkl *.png "$backup_dir/" 2>/dev/null
        cp dataset_WIFI7.csv "$backup_dir/" 2>/dev/null
        echo "✅ Backup created: $backup_dir"
        ls -lh "$backup_dir/"
        ;;
    
    info)
        echo "ℹ️  Antenna ML Project Info"
        echo "================================"
        echo "Container: antenna_ml"
        echo "Working dir: /app"
        echo "Port: 7860 (Gradio)"
        echo ""
        echo "📊 Container status:"
        sudo docker-compose ps
        echo ""
        echo "📁 Project files:"
        ls -lh
        echo ""
        echo "💿 Disk usage:"
        sudo docker system df
        ;;
    
    help|--help|-h|"")
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║          🛜 Antenna ML Docker Helper Script               ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo ""
        echo "📋 Usage: ./run.sh [command]"
        echo ""
        echo "🐳 Container Management:"
        echo "  start       Start the Docker container"
        echo "  stop        Stop the Docker container"
        echo "  restart     Restart the container"
        echo "  rebuild     Rebuild container (after changing Dockerfile/requirements)"
        echo "  status      Show container status and output files"
        echo "  logs        View container logs (Ctrl+C to stop)"
        echo "  info        Show detailed project info"
        echo ""
        echo "🤖 Machine Learning:"
        echo "  train       Train the antenna prediction model"
        echo "  tune        Run hyperparameter tuning (5-10 min)"
        echo "  test        Run the test suite (pytest + coverage)"
        echo "  app         Start Gradio web interface (http://localhost:7860)"
        echo "  build-exe   Build standalone executable with PyInstaller"
        echo ""
        echo "🔧 Development:"
        echo "  shell       Open bash shell in container"
        echo "  python      Open Python shell in container"
        echo "  exec <cmd>  Execute custom command in container"
        echo ""
        echo "📁 File Management:"
        echo "  upload <file>    Upload file to container"
        echo "  download <file>  Download file from container"
        echo "  backup           Create timestamped backup of outputs"
        echo ""
        echo "🧹 Cleanup:"
        echo "  clean       Remove output files (.pkl, .png)"
        echo "  clean-all   Remove containers, images, and outputs"
        echo "  update      Update Python packages (rebuild required)"
        echo ""
        echo "📚 Examples:"
        echo "  ./run.sh start"
        echo "  ./run.sh train"
        echo "  ./run.sh app"
        echo "  ./run.sh exec 'ls -la'"
        echo "  ./run.sh upload new_dataset.csv"
        echo "  ./run.sh backup"
        echo ""
        ;;
    
    *)
        echo "❌ Unknown command: $1"
        echo "Run './run.sh help' to see available commands"
        exit 1
        ;;
esac
