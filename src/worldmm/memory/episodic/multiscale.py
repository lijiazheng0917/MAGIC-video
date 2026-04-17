from egorag.agents.RagAgent import RagAgent
from egorag.database.Chroma import Chroma
from egorag.utils.gen_event import gen_event


def generate_multiscale_memory(db_name: str = "A1_JAKE", 
                          json_path: str = "data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_30sec.json",
                          diary_dir: str = ".cache/events_diary",
                          save_path: str = "data/EgoLife/EgoLifeCap"):
    """
    Run the multiscale episodic memory pipeline.
    
    Args:
        db_name: Name of the Chroma database
        json_path: Path to the JSON file for database creation
        diary_dir: Path to the diary directory
        save_path: Path to save the generated events
    """
    # Initialize the Chroma database with the provided name
    db_t = Chroma(name=db_name)

    # Initialize the RagAgent
    agent = RagAgent(database_t=db_t, name=db_name, video_base_dir=f"data/EgoLife/{db_name}")

    # Create the database from the provided JSON file
    agent.create_database_from_json(json_path)

    gen_event(db_name, diary_dir, save_path)


# Entry point for the script
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize RagAgent with database and JSON path."
    )
    parser.add_argument("--db_name", default="A1_JAKE", help="Name of the Chroma database")
    parser.add_argument("--json_path", default="data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_30sec.json", help="Path to the JSON file for database creation")
    parser.add_argument("--diary_dir", default=".cache/events_diary", help="Path to the diary directory")
    parser.add_argument("--save_path", default="data/EgoLife/EgoLifeCap", help="Path to save the generated events")

    args = parser.parse_args()

    generate_multiscale_memory(
        db_name=args.db_name,
        json_path=args.json_path,
        diary_dir=args.diary_dir,
        save_path=args.save_path
    )
