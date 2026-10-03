"""테스트용 가짜 transformers (dry-run 토크나이저만)."""
__version__ = "mock"


class AutoTokenizer:
    @staticmethod
    def from_pretrained(path, **kw):
        from vllm import _Tok
        return _Tok()
