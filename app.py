# app.py

from rag_pipeline.vector_store import build_vector_store, load_vector_store
from rag_pipeline.embeddings import get_embedding_model
from memory_store import (
    load_memory, save_memory,
    add_message, get_history,
    clear_scratchpad
)
from profile_store import load_profile, save_profile
from agent.agent import run_react_agent


def process_message(user_id: str, query: str) -> str:
    try:
        # Load short-term memory — current conversation
        session = load_memory(user_id)
        history = get_history(session)

        # Load long-term memory — purchase history only
        profile = load_profile(user_id)

        # Run agent — profile passed for optional context injection
        response = run_react_agent(
            user_query=query,
            user_id=user_id,
            history=history,
            profile=profile
        )

        # Save short-term memory
        add_message(session, "user",      query)
        add_message(session, "assistant", response)
        clear_scratchpad(session)
        save_memory(user_id, session)

        # Save long-term memory
        # (profile may have been updated — e.g. new purchase recorded)
        save_profile(user_id, profile)

        return response

    except Exception as e:
        print(f"[PIPELINE ERROR] {e}")
        return "Something went wrong. Please try again."


if __name__ == '__main__':
    print('Loading embedding model')
    get_embedding_model()
    print('Embedding model loaded.')

    index, documents = load_vector_store()
    if index is None:
        print('Building vector store...')
        build_vector_store()

    print('Agent is online. Type (exit) to quit.\n')

    try:
        while True:
            user_input = input('User: ')
            if user_input.lower() == 'exit':
                print('Goodbye!')
                break
            response = process_message('test_user', user_input)
            print(f'Agent: {response}')
            print('\n' + '-'*50 + '\n')

    except KeyboardInterrupt:
        print('\nGoodbye!')