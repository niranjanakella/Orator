"""
Pure G2P (Grapheme-to-Phoneme) module for Orator TTS Engine

This module provides text-to-phoneme conversion without any PyTorch dependency.
It uses espeak-ng for phoneme generation with IPA output.
"""

import logging
import re
import subprocess
import shutil
from typing import Generator, Tuple, List, Optional

logger = logging.getLogger(__name__)

# Language code mappings
LANG_CODES = {
    'a': 'en-us',  # American English
    'b': 'en-gb',  # British English
    'e': 'es',     # Spanish
    'f': 'fr-fr',  # French
    'h': 'hi',     # Hindi
    'i': 'it',     # Italian
    'p': 'pt-br',  # Portuguese
    'j': 'ja',     # Japanese
    'z': 'zh',     # Chinese
}

# Custom pronunciations for specific words
CUSTOM_LEXICON = {
    'a': {  # American English
        'kokoro': 'kˈOkəɹO',
    },
    'b': {  # British English
        'kokoro': 'kˈQkəɹQ',
    },
}

# IPA character mappings for normalization (espeak output -> model expected)
# Based on the tokenizer vocabulary
IPA_NORMALIZATIONS = {
    # Common substitutions for espeak output compatibility
    'ɚ': 'ɚ',
    'ɝ': 'ɚ',
    'ɾ': 'ɾ',
    'ʔ': 'ʔ',
    'ː': 'ː',
    'ˈ': 'ˈ',
    'ˌ': 'ˌ',
}


class EspeakG2P:
    """
    Grapheme-to-Phoneme converter using espeak-ng.
    
    Converts text to IPA phonemes compatible with the Kokoro TTS model.
    No PyTorch dependency required.
    """
    
    def __init__(self, lang_code: str = 'a'):
        """
        Initialize the G2P converter.
        
        Args:
            lang_code: Single character language code ('a' for American English, 'b' for British, etc.)
        """
        self.lang_code = lang_code
        self.espeak_lang = LANG_CODES.get(lang_code, 'en-us')
        self.custom_lexicon = CUSTOM_LEXICON.get(lang_code, {})
        
        # Check if espeak-ng is available
        self.espeak_path = self._find_espeak()
        if not self.espeak_path:
            raise RuntimeError(
                "espeak-ng not found. Please install it:\n"
                "  macOS: brew install espeak-ng\n"
                "  Linux: apt-get install espeak-ng\n"
                "  Windows: Download from https://github.com/espeak-ng/espeak-ng/releases"
            )
        
        logger.info(f"EspeakG2P initialized with language: {self.espeak_lang}")
    
    def _find_espeak(self) -> Optional[str]:
        """Find espeak-ng executable path."""
        # Try common paths
        paths_to_try = [
            'espeak-ng',
            '/opt/homebrew/bin/espeak-ng',
            '/usr/local/bin/espeak-ng',
            '/usr/bin/espeak-ng',
        ]
        
        for path in paths_to_try:
            if shutil.which(path):
                return path
        
        return None
    
    def _apply_custom_lexicon(self, text: str) -> str:
        """Apply custom pronunciations for specific words."""
        # This is a placeholder - in a real implementation, we'd need to
        # mark these words for special handling
        return text
    
    def _run_espeak(self, text: str) -> str:
        """
        Run espeak-ng to get IPA phonemes.
        
        Args:
            text: Input text to convert
            
        Returns:
            IPA phoneme string
        """
        try:
            # Use espeak-ng with IPA output
            # -q: quiet (no sound)
            # --ipa: output IPA phonemes
            # -v: voice/language
            result = subprocess.run(
                [self.espeak_path, '-q', '--ipa', '-v', self.espeak_lang, text],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.warning(f"espeak-ng returned non-zero: {result.stderr}")
                return ""
            
            phonemes = result.stdout.strip()
            
            # Clean up the output
            phonemes = self._normalize_phonemes(phonemes)
            
            return phonemes
            
        except subprocess.TimeoutExpired:
            logger.error("espeak-ng timed out")
            return ""
        except Exception as e:
            logger.error(f"espeak-ng error: {e}")
            return ""
    
    def _normalize_phonemes(self, phonemes: str) -> str:
        """
        Normalize phoneme output for model compatibility.
        
        Args:
            phonemes: Raw IPA phonemes from espeak
            
        Returns:
            Normalized phoneme string
        """
        # Remove newlines and extra whitespace
        phonemes = ' '.join(phonemes.split())
        
        # Apply IPA normalizations
        for old, new in IPA_NORMALIZATIONS.items():
            phonemes = phonemes.replace(old, new)
        
        # Remove any characters that aren't in our vocabulary
        # Keep only valid IPA characters and punctuation
        
        return phonemes
    
    def phonemize(self, text: str) -> str:
        """
        Convert text to phonemes.
        
        Args:
            text: Input text
            
        Returns:
            Phoneme string
        """
        if not text or not text.strip():
            return ""
        
        # Apply custom lexicon
        text = self._apply_custom_lexicon(text)
        
        # Get phonemes from espeak
        phonemes = self._run_espeak(text)
        
        return phonemes
    
    def __call__(self, text: str) -> str:
        """Shorthand for phonemize()."""
        return self.phonemize(text)


class PureG2PPipeline:
    """
    Pure G2P Pipeline for text processing without PyTorch.
    
    Handles:
    - Text chunking at sentence boundaries
    - Phoneme generation via espeak-ng
    - Phoneme length limiting (max 510 for model)
    """
    
    MAX_PHONEME_LENGTH = 510
    
    def __init__(self, lang_code: str = 'a'):
        """
        Initialize the G2P pipeline.
        
        Args:
            lang_code: Single character language code ('a' for American English, etc.)
        """
        self.lang_code = lang_code
        self.g2p = EspeakG2P(lang_code)
        self.custom_lexicon = CUSTOM_LEXICON.get(lang_code, {})
        
        logger.info(f"PureG2PPipeline initialized for language: {lang_code}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _chunk_text(self, text: str, max_chars: int = 400) -> List[str]:
        """
        Split text into chunks suitable for processing.
        
        Args:
            text: Input text
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return [text] if text.strip() else []
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If sentence itself is too long, split it
                if len(sentence) > max_chars:
                    # Split on commas or other breaks
                    parts = re.split(r'(?<=[,;:])\s+', sentence)
                    sub_chunk = ""
                    for part in parts:
                        if len(sub_chunk) + len(part) + 1 <= max_chars:
                            sub_chunk = (sub_chunk + " " + part).strip()
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = part
                    if sub_chunk:
                        current_chunk = sub_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _apply_custom_words(self, text: str, phonemes: str) -> str:
        """
        Apply custom pronunciations for known words.
        
        Args:
            text: Original text
            phonemes: Generated phonemes
            
        Returns:
            Phonemes with custom pronunciations applied
        """
        text_lower = text.lower()
        
        for word, pronunciation in self.custom_lexicon.items():
            if word in text_lower:
                # This is a simplified approach - for production,
                # we'd need more sophisticated word boundary detection
                logger.debug(f"Custom pronunciation applied for: {word}")
        
        return phonemes
    
    def process(
        self, 
        text: str, 
        voice: str = None,
        speed: float = 1.0
    ) -> Generator[Tuple[str, str, None], None, None]:
        """
        Process text and yield (graphemes, phonemes, None) tuples.
        
        This matches the KPipeline interface but without audio generation.
        
        Args:
            text: Input text to process
            voice: Voice name (used for language detection from first character)
            speed: Speech speed (not used for G2P, passed for compatibility)
            
        Yields:
            Tuple of (graphemes, phonemes, None)
        """
        if not text or not text.strip():
            return
        
        # Split text into newline-separated segments first
        segments = re.split(r'\n+', text.strip())
        
        for segment in segments:
            if not segment.strip():
                continue
            
            # Chunk the segment
            chunks = self._chunk_text(segment)
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                
                # Generate phonemes
                phonemes = self.g2p.phonemize(chunk)
                
                if not phonemes:
                    continue
                
                # Apply custom pronunciations
                phonemes = self._apply_custom_words(chunk, phonemes)
                
                # Truncate if too long
                if len(phonemes) > self.MAX_PHONEME_LENGTH:
                    logger.warning(
                        f"Phonemes too long ({len(phonemes)} > {self.MAX_PHONEME_LENGTH}), truncating"
                    )
                    phonemes = phonemes[:self.MAX_PHONEME_LENGTH]
                
                yield (chunk, phonemes, None)
    
    def __call__(
        self, 
        text: str, 
        voice: str = None,
        speed: float = 1.0
    ) -> Generator[Tuple[str, str, None], None, None]:
        """Shorthand for process()."""
        yield from self.process(text, voice, speed)

