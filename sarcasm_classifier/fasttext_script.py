import fasttext

class SarcasmService(object):
    model = None

    @classmethod
    def load_model(self):
        loaded_model = fasttext.load_model('fasttext_sarcasm.ftz')
        return loaded_model

    @classmethod
    def get_model(self):
        if self.model is None:
            self.model = self.load_model()
        return self.model


def predict_is_sarcastic(text):
    return SarcasmService.get_model().predict(text, k=2)

if __name__ == '__main__':
    ip = 'Make Indian manufacturing competitive to curb Chinese imports: RC Bhargava'
    print(f'Result : {predict_is_sarcastic(ip)}')