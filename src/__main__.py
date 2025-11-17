import typer
from src.cli.train import app as train_app
from src.cli.mean_sim import app as mean_sim_app
from src.cli.matrix_sim import app as matrix_sim_app
from src.cli.cluster_proto import app as cluster_proto_app

app = typer.Typer(help="Outils d'analyse des relations génitives")



if __name__ == "__main__":
    app()