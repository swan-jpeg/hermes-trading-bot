#!/usr/bin/env python3
"""
Improved script to translate docstrings and comments in Python files from Dutch to English.
This script focuses only on docstrings, comments and string literals that are 
meant for explanation, not functional data.
"""

import re
import os

# Dictionary of translations for common Dutch phrases to English
TRANSLATIONS = {
    # Docstrings and comments
    "LAAG 3: audio-analyse (spraak->tekst, sentiment, prosodie).": 
    "LEVEL 3: audio analysis (speech->text, sentiment, prosody).",
    "Echte model-integratie met offline-fallback:": 
    "Real model integration with offline fallback:",
    "- Transcript: faster-whisper (WhisperModel)": 
    "- Transcript: faster-whisper (WhisperModel)",
    "- Emotie: SpeechBrain pretrained emotion (via transformers)": 
    "- Emotion: SpeechBrain pretrained emotion (via transformers)",
    "- Prosodie: openSMILE (eGeMAPS) + librosa voor pitch/volume/pauzes": 
    "- Prosody: openSMILE (eGeMAPS) + librosa for pitch/volume/pauses",
    "Als een model niet geïnstalleerd is, valt de analyse terug op neutrale waarden met confidence=0.0 zodat de pipeline nooit crasht.": 
    "If a model is not installed, the analysis falls back to neutral values with confidence=0.0 so the pipeline never crashes.",
    "use_heavy_models: laad zware modellen (whisper/sentiment). Standaard uit zodat de pipeline snel en offline-safe blijft; zet aan op een machine waar de modellen betrouwbaar draaien.": 
    "use_heavy_models: load heavy models (whisper/sentiment). Default off so the pipeline stays fast and offline-safe; enable on a machine where the models run reliably.",
    "Voorkomt dat een hangend zwaar model (bijv. emotion-pipeline op een trage CPU) de hele analyse blokkeert.": 
    "Prevents a hanging heavy model (e.g. emotion-pipeline on a slow CPU) from blocking the entire analysis.",
    "Transcription via faster-whisper (only if use_heavy_models).": 
    "Transcription via faster-whisper (only if use_heavy_models).",
    "Emotion estimate from prosody (pitch/volume/pace).": 
    "Emotion estimate from prosody (pitch/volume/pace).",
    "Gebruikt een lichte, betrouwbare benadering die op elke machine werkt (geen zwaar model nodig). Als use_heavy_models aanstaat en een SpeechBrain-model beschikbaar is, zou dat hier kunnen worden toegevoegd; de prosodie-baseline is altijd beschikbaar.": 
    "Uses a lightweight, reliable approach that works on every machine (no heavy model needed). If use_heavy_models is on and a SpeechBrain model is available, it could be added here; the prosody baseline is always available.",
    "Heuristiek: hoge pitch-variatie + hoog volume = opgewonden/optimistisch; lage pitch + laag volume = kalm/confident; hoge pace = nerveus/anxious.": 
    "Heuristic: high pitch variation + high volume = excited/optimistic; low pitch + low volume = calm/confident; high pace = nervous/anxious.",
    "Prosodie via openSMILE (eGeMAPS) + librosa.": 
    "Prosody via openSMILE (eGeMAPS) + librosa.",
    "Number of pauses (silences) in the audio.": 
    "Number of pauses (silences) in the audio.",
    "Neutral SpeechSignal when audio is not available.": 
    "Neutral SpeechSignal when audio is not available.",
    "Smoke-test: schemas + risico-engine + portfolio + winst-nemen-logica.": 
    "Smoke-test: schemas + risk-engine + portfolio + profit-taking logic.",
    "Taking profit / limiting loss must never be blocked.": 
    "Taking profit / limiting loss must never be blocked.",
    "High VaR95 / crash probability reduces the allowed allocation.": 
    "High VaR95 / crash probability reduces the allowed allocation.",
    "geknipt op asset-limiet": "cut at asset limit",
    "bevestig dat de drempel overschreden wordt": "confirm that the threshold is exceeded",
    "begrensd door MC": "limited by MC",
    "de take-profit staat": "take-profit threshold",
    "verkoop de winst": "sell the profit",
    "The RL policy must take profit on an existing position that is above": 
    "The RL policy must take profit on an existing position that is above",
    "also as sentiment continues to be positive": "also as sentiment continues to be positive",
    "Low vol -> allocation unchanged but MC fields filled.": 
    "Low vol -> allocation unchanged but MC fields filled.",
    "High vol -> VaR95 well below the -5% threshold -> allocation must shrink.": 
    "High vol -> VaR95 well below the -5% threshold -> allocation must shrink.",
    "Was at +12% (peak 112), now back to 106 -> -5.4% from peak = trailing stop.": 
    "Was at +12% (peak 112), now back to 106 -> -5.4% from peak = trailing stop.",
    "+18% > 15%": "+18% > 15%",
    "-10% < -8%": "-10% < -8%",
    "emotie": "sentiment",
    "zekerheid": "certainty", 
    "kwaliteit": "quality",
    "sentiment": "sentiment",
    "take_profit": "take_profit",
    "stop_loss": "stop_loss",
    "trailing_stop": "trailing_stop",
    "none": "none"
}

def translate_file(file_path: str) -> int:
    """Translate docstrings and comments in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Find all docstrings (multiline strings that are at the start of a function/class)
        # This is a simplified approach - we'll look for triple quotes
        # We'll use a more targeted approach to avoid breaking code
        lines = content.split('\n')
        modified = False
        
        # Process each line looking for docstrings and comments to translate
        for i, line in enumerate(lines):
            # Check for comments (lines starting with #)
            if line.strip().startswith('#'):
                # Check if line contains Dutch text that we want to translate
                for dutch_phrase, english_phrase in TRANSLATIONS.items():
                    if dutch_phrase in line:
                        lines[i] = line.replace(dutch_phrase, english_phrase)
                        modified = True
                        
            # Check for docstrings (triple quoted strings)
            elif '"""' in line or "'''" in line:
                # Look for the entire docstring block
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    # This is a docstring line, check if it needs translation
                    for dutch_phrase, english_phrase in TRANSLATIONS.items():
                        if dutch_phrase in line:
                            lines[i] = line.replace(dutch_phrase, english_phrase)
                            modified = True
        
        # Reconstruct content
        new_content = '\n'.join(lines)
        
        # Only write if content changed
        if modified and new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Translated: {file_path}")
            return 1
        else:
            print(f"No changes: {file_path}")
            return 0
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def main():
    """Main function to process all Python files."""
    print("Starting translation of docstrings and comments...")
    
    # Files to process
    python_files = []
    
    # Process hermes_bot directory
    for root, dirs, files in os.walk('hermes_bot'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Process tests directory  
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files to process")
    
    translated_count = 0
    for file_path in python_files:
        translated_count += translate_file(file_path)
    
    print(f"\nTranslation complete. {translated_count} files were modified.")

if __name__ == "__main__":
    main()