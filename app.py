import streamlit as st
from src.main import (
    process_document,
    process_user_query,
)  # Importiere die Funktionen aus main.py

# Streamlit UI
st.title("PDF-basierter Chatbot")
st.write("Lade mehrere Dokumente hoch, um Fragen dazu zu stellen.")

# Dokumentenhochladung (mehrere Dateien)
uploaded_files = st.file_uploader(
    "Lade PDF-Dokumente hoch", type=["pdf"], accept_multiple_files=True
)

# Überprüfen, ob Dokumente hochgeladen wurden
if uploaded_files:
    # Speichern der hochgeladenen Dateien in der Session
    st.session_state.uploaded_files = uploaded_files

    # Text anzeigen, während die Dokumente verarbeitet werden
    with st.spinner("Verarbeite Dokumente..."):
        if "document_data" not in st.session_state:
            st.session_state.document_data = []
            # Verarbeite alle hochgeladenen Dokumente
            for uploaded_file in uploaded_files:
                # Verarbeite das Dokument und speichere die Daten
                st.session_state.document_data.append(process_document(uploaded_file))

        st.success("Dokumente erfolgreich hochgeladen und verarbeitet!")

# Benutzeranfrage (Text-Eingabe)
user_query = st.text_input("Stelle eine Frage zum Dokument:")

if user_query:
    with st.spinner("Suche nach relevanten Informationen..."):
        # Generiere Antwort basierend auf der Benutzeranfrage
        response = process_user_query(
            user_query
        )  # Verwende die Funktion aus main.py, um die Anfrage zu bearbeiten

    # Zeige Antwort
    st.write("Antwort:", response)
