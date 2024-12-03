# from graphviz import Digraph

# # Create a new directed graph
# dag = Digraph(format='png', engine='dot')

# # Add nodes
# dag.node("A", "Data Ingestion")
# dag.node("B", "Data Cleaning")
# dag.node("C", "Feature Engineering")
# dag.node("D", "Model Training")
# dag.node("E", "Model Inference")
# dag.node("F", "Database Storage")
# dag.node("G", "Visualization")
# dag.node("H", "User Interaction")

# # Add edges
# dag.edges([
#     ("A", "B"),  # Ingestion → Cleaning
#     ("B", "C"),  # Cleaning → Feature Engineering
#     ("C", "D"),  # Feature Engineering → Training
#     ("D", "E"),  # Training → Inference
#     ("E", "F"),  # Inference → Storage
#     ("F", "G"),  # Storage → Visualization
#     ("G", "H")   # Visualization → User Interaction
# ])

# # Render graph
# dag.render("project_dag", view=True)

# from airflow import DAG
# from airflow.operators.dummy_operator import DummyOperator
# from datetime import datetime

# dag = DAG(
#     'project_workflow',
#     default_args={'start_date': datetime(2024, 12, 1)},
#     schedule_interval=None,
# )

# ingestion = DummyOperator(task_id='data_ingestion', dag=dag)
# cleaning = DummyOperator(task_id='data_cleaning', dag=dag)
# feature_eng = DummyOperator(task_id='feature_engineering', dag=dag)
# training = DummyOperator(task_id='model_training', dag=dag)
# inference = DummyOperator(task_id='model_inference', dag=dag)
# storage = DummyOperator(task_id='database_storage', dag=dag)
# visualization = DummyOperator(task_id='visualization', dag=dag)
# interaction = DummyOperator(task_id='user_interaction', dag=dag)

# # Define dependencies
# ingestion >> cleaning >> feature_eng >> training >> inference >> storage >> visualization >> interaction

