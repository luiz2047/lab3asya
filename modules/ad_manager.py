class AdManager:
    def __init__(self):
        self.ad_intents = [2]
        self.ads = {
            2: 'Check out our latest products: Product A, Product B, and Product C!'
        }

    def check_for_ad(self, intent):
        if intent in self.ad_intents:
            return self.ads[intent]
        return ''
