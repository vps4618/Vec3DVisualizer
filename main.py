from fastapi import FastAPI
from pydantic import BaseModel
import plotly.graph_objects as go
import numpy as np

app = FastAPI()

# 1. Define the structure of the data we expect from the MAUI app
class MathPayload(BaseModel):
    mode : str # Can be "plane" , "line" , "skew" or "angle"

    # We use lists to represent vectors/ points [x,y,z] or [a,b,c]
    vector_a : list[float] = [0,0,0]
    vector_b: list[float] = [0,0,0]
    point_1: list[float] = [0,0,0]
    point_2 : list[float] = [0,0,0]

# 2. The single API endpoint that processes all requests
@app.post("/api/plot")
def generate_plot(payload : MathPayload):
    fig = go.Figure()

    if payload.mode == "plane":
        # Equation : ax + by + cz = d
        a, b, c, d = payload.vector_a[0] , payload.vector_a[1] , payload.vector_a[2], payload.vector_a[3]
        # linespace give list of points in that interval
        x = np.linspace(-10 , 10 , 20)
        y = np.linspace(-10 , 10 , 20)
        # Return a tuple of coordinate matrices from coordinate vectors.
        X , Y = np.meshgrid(x,y)
        Z = (d - a*X - b*Y) / c # calculate z values

        fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=0.8))
        fig.update_layout(title = f"Plane: {a}x + {b}y + {c}z = {d}")

    elif payload.mode == "line":
        # Equation: r  = r0 + t * v
        r0 = np.array(payload.point_1)
        v = np.array(payload.vector_a)

        # Generate t values from -10 to 10 (100 values)
        t_values = np.linspace(-10,10,100)

        # Calculate points:x, y , z for each t
        # <x,y,z> = <x1,y1,z1> + t <a1,b1,c1>
        points = np.array([r0 + t * v for t in t_values])

        fig.add_trace(go.Scatter3d(
            x=points[:,0] , y=points[:,1], z=points[:,2],
            mode = 'lines', line=dict(color='red', width =5)
        ))
        fig.update_layout(title="Vector Equation of a Line")

    elif payload.mode == "skew":
        # Skew lines logic
        A = np.array(payload.point_1)
        B = np.array(payload.point_2)
        v0 = np.array(payload.vector_a)
        v1 = np.array(payload.vector_b)

        # Draw Line 1 (r = A + t*v0)
        t_vals = np.linspace(-10, 10, 50)
        L1 = np.array([A + t * v0 for t in t_vals])
        fig.add_trace(go.Scatter3d(x=L1[:, 0], y=L1[:, 1], z=L1[:, 2], mode='lines', name='Line 1'))

        # Draw Line 2 (r = B + t*v1)
        L2 = np.array([B + t * v1 for t in t_vals])
        fig.add_trace(go.Scatter3d(x=L2[:, 0], y=L2[:, 1], z=L2[:, 2], mode='lines', name='Line 2'))

        # Calculate shortest distance: |AB dot n| / |n|
        AB = B - A
        n = np.cross(v0, v1)  # The cross product gives the normal vector to both lines
        n_mag = np.linalg.norm(n)

        if n_mag != 0:
            distance = abs(np.dot(AB, n)) / n_mag
            # Add text to the graph displaying the calculated distance
            fig.add_annotation(text=f"Shortest Distance of skew lines: {distance:.2f}",
                               xref="paper", yref="paper", x=0, y=1, showarrow=False)

        # 3. Finalize the layout and convert to HTML
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')

    return {"status": "success", "html": html_content}