# app.py

from rag_pipeline.vector_store import build_vector_store, load_vector_store
from rag_pipeline.embeddings import get_embedding_model
from memory_store import (
    load_memory, save_memory,
    add_message, get_history,
    clear_scratchpad
)
from profile_store import load_profile, save_profile
from orchestrator import run_orchestrator


def process_message(user_id: str, query: str) -> str:
    try:
        # Load short-term memory
        session = load_memory(user_id)
        history = get_history(session)

        # Load long-term memory
        profile = load_profile(user_id)

        # Run orchestrator — handles classification + routing
        response = run_orchestrator(
            user_query=query,
            user_id=user_id,
            session=session,
            history=history,
            profile=profile
        )

        # Save memories
        add_message(session, "user",      query)
        add_message(session, "assistant", response)
        clear_scratchpad(session)
        save_memory(user_id, session)
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