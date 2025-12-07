"""Musical chapter end indicators.

This module generates synthetic musical tones to mark the end of chapters
during audiobook creation. Each indicator is a composition of two randomly
selected musical events.
"""

import random

import numpy as np


class ChapterIndicatorGenerator:
    """Generator for musical chapter end indicators."""

    def __init__(self, sample_rate: int = 24000):
        """Initialize the generator.

        Args:
            sample_rate: Audio sample rate in Hz (default: 24000 for Kokoro)
        """
        self.sample_rate = sample_rate

    def _envelope(
        self, t: np.ndarray, attack: float = 0.01, release: float = 0.2
    ) -> np.ndarray:
        """Simple fade-in/fade-out envelope.

        Args:
            t: Time array
            attack: Attack time in seconds
            release: Release time in seconds

        Returns:
            Envelope array
        """
        env = np.ones_like(t)
        attack_n = int(self.sample_rate * attack)
        release_n = int(self.sample_rate * release)
        env[:attack_n] *= np.linspace(0, 1, attack_n)
        env[-release_n:] *= np.linspace(1, 0, release_n)
        return env

    def _normalize(self, sig: np.ndarray) -> np.ndarray:
        """Normalize signal to [-1, 1] range.

        Args:
            sig: Input signal

        Returns:
            Normalized signal
        """
        max_val = np.max(np.abs(sig))
        if max_val > 0:
            return sig / max_val
        return sig

    # ------------ Musical Building Blocks ------------

    def arpeggio(
        self,
        notes: list[float] | None = None,
        tone_len: float = 0.25,
        gap: float = 0.03,
    ) -> np.ndarray:
        """Classical upward arpeggio.

        Args:
            notes: List of frequencies in Hz (default: C major arpeggio)
            tone_len: Duration of each note in seconds
            gap: Gap between notes in seconds

        Returns:
            Audio signal
        """
        if notes is None:
            notes = [523.25, 659.25, 783.99, 1046.5]  # C, E, G, C

        pieces = []
        for f in notes:
            t = np.linspace(0, tone_len, int(self.sample_rate * tone_len), False)
            sig = np.sin(2 * np.pi * f * t) * self._envelope(t)
            pieces.append(sig)
            pieces.append(np.zeros(int(self.sample_rate * gap)))
        return self._normalize(np.concatenate(pieces))

    def descending_arpeggio(self) -> np.ndarray:
        """Downward classical arpeggio.

        Returns:
            Audio signal
        """
        return self.arpeggio(notes=[1046.5, 783.99, 659.25, 523.25])

    def pizzicato_pop(self) -> np.ndarray:
        """A short pizzicato pluck-like sound.

        Returns:
            Audio signal
        """
        duration = 0.25
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        f = 440  # A4
        sig = np.sin(2 * np.pi * f * t) * np.exp(-15 * t)
        sig *= self._envelope(t, attack=0.005, release=0.1)
        return self._normalize(sig)

    def gliss_up(self) -> np.ndarray:
        """Harp-like upward glissando.

        Returns:
            Audio signal
        """
        duration = 0.5
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        freqs = np.linspace(400, 1600, len(t))
        sig = np.sin(2 * np.pi * freqs * t) * np.exp(-2 * t)
        return self._normalize(sig)

    def bell_chime(self) -> np.ndarray:
        """Classical bell/chime sound.

        Returns:
            Audio signal
        """
        duration = 1.2
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        freqs = [880, 1310, 1980, 2450]
        decays = [2.5, 3.2, 4.0, 5.0]
        sig = np.zeros_like(t)
        for f, d in zip(freqs, decays, strict=True):
            sig += np.sin(2 * np.pi * f * t) * np.exp(-d * t)
        sig *= self._envelope(t, attack=0.005, release=0.4)
        return self._normalize(sig)

    def cadence(self) -> np.ndarray:
        """Simple classical I–V–I cadence.

        Returns:
            Audio signal
        """
        notes = [523.25, 392.00, 523.25]  # C – G – C
        tone_len = 0.35
        gap = 0.05
        pieces = []
        for f in notes:
            t = np.linspace(0, tone_len, int(self.sample_rate * tone_len), False)
            sig = np.sin(2 * np.pi * f * t) * self._envelope(t)
            pieces.append(sig)
            pieces.append(np.zeros(int(self.sample_rate * gap)))
        return self._normalize(np.concatenate(pieces))

    # ------------ Random Selection ------------

    def generate(self, verbose: bool = False) -> np.ndarray:
        """Generate a random chapter end indicator.

        Combines two randomly selected musical events with a gap between them.

        Args:
            verbose: If True, print which events were selected

        Returns:
            Audio signal as float32 numpy array
        """
        # First event options
        first_options = {
            "arpeggio_up": self.arpeggio,
            "arpeggio_down": self.descending_arpeggio,
            "pizzicato": self.pizzicato_pop,
            "gliss_up": self.gliss_up,
        }

        # Second event options
        second_options = {
            "bell": self.bell_chime,
            "cadence": self.cadence,
            "pizzicato": self.pizzicato_pop,
        }

        # Select and generate first event
        name1 = random.choice(list(first_options.keys()))
        part1 = first_options[name1]()

        # Select and generate second event
        name2 = random.choice(list(second_options.keys()))
        part2 = second_options[name2]()

        # Combine with gap
        gap = np.zeros(int(self.sample_rate * 0.25))
        final = self._normalize(np.concatenate([part1, gap, part2]))

        if verbose:
            print("Generated chapter indicator:")
            print(f"  First:  {name1}")
            print(f"  Second: {name2}")

        # Return as float32
        return final.astype(np.float32)
