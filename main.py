from agents.preprocessing_agent import graph_builder, state_initialiser
import asyncio

state = state_initialiser('sample_inputs/job_descriptions.csv', 'sample_inputs/Advait-Menon_May_2025_Resume.docx')
agent = graph_builder()
print("Running the preprocessing agent")
res = asyncio.run(agent.ainvoke(state))
print(res['relevant_job_fields'])
print(res['mapped_resume_fields_to_job_fields'])