from flask import Flask
import sys

from app.routes.dashboard_routes import frontend_dashboard # adjust path if needed
from app.routes.dashboard_api_routes import main_bp # adjust path if needed

app = Flask(__name__)
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')  #add template folder path

app.register_blueprint(frontend_dashboard, url_prefix='/')
app.register_blueprint(main_bp, url_prefix='/api')

if __name__ == '__main__':
    #app.run(debug=True)
    host = "127.0.0.1"
    port = 5000
    if "--host" in sys.argv:
        host = sys.argv[sys.argv.index("--host")+1]
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port")+1])
    app.run(debug=True, host=host, port=port)