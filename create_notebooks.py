
import nbformat as nbf
import os

def create_notebook(script_path, notebook_path):
    with open(script_path, 'r') as f:
        code = f.read()
    
    nb = nbf.v4.new_notebook()
    
    # Add a markdown cell with the filename as title
    nb['cells'].append(nbf.v4.new_markdown_cell(f"# {os.path.basename(script_path)}"))
    
    # Add the code cell
    nb['cells'].append(nbf.v4.new_code_cell(code))
    
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    source_dir = "source"
    notebook_dir = "notebooks"
    
    if not os.path.exists(notebook_dir):
        os.makedirs(notebook_dir)
        
    files = [
        "Dynamic_Scale_Simulation.py",
        "Emergence_Omega_Sim.py",
        "Horizon_Shredding_Model.py",
        "Informational_Geometry_Test.py",
        "Sim5_Emergent_Gravity.py",
        "sim6_v16_omega.py"
    ]
    
    for f in files:
        script_path = os.path.join(source_dir, f)
        notebook_path = os.path.join(notebook_dir, f.replace(".py", ".ipynb"))
        if os.path.exists(script_path):
            create_notebook(script_path, notebook_path)
            print(f"Created {notebook_path}")
        else:
            print(f"Script not found: {script_path}")
