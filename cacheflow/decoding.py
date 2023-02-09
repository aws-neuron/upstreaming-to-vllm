from typing import List


class DecodingParams:

    def __init__(
        self,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        use_beam_search: bool = False,
        stop_token_ids: List[int] = [],
    ) -> None:
        assert n >= 1
        assert temperature >= 0.0
        assert 0.0 < top_p <= 1.0
        if use_beam_search:
            assert n > 1
            assert temperature > 0.0
            assert top_p == 1.0
        elif temperature == 0.0:
            # Zero temperature means greedy decoding.
            assert n == 1
            assert top_p == 1.0

        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.use_beam_search = use_beam_search
        self.stop_token_ids = stop_token_ids
