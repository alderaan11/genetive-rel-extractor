import typer
from src2.cli.train import app as train_app
from src2.cli.mean_sim import app as mean_sim_app
from src2.cli.matrix_sim import app as matrix_sim_app
from src2.cli.cluster_proto import app as cluster_proto_app

app = typer.Typer(help="Outils d'analyse des relations génitives")

app.add_typer(train_app, name="train-genitif")
app.add_typer(mean_sim_app, name="mean-sim")
app.add_typer(matrix_sim_app, name="matrix-sim")
app.add_typer(cluster_proto_app, name="cluster-proto")

if __name__ == "__main__":
    app()