from flask import Flask, render_template, jsonify
import subprocess
import sys

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/run_script")
def run_script():
    try:
        # Use sys.executable to ensure Python version consistency
        output = subprocess.check_output([sys.executable, "script.py"], universal_newlines=True)
        return jsonify({"success": True, "output": output})
    except subprocess.CalledProcessError as e:
        return jsonify({"success": False, "error": e.output}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
