import re
class CallBack:
    _order=0
    
    def set_algorithm(self, alg): self.alg=alg
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return name or 'callback'


class PrintMutation(CallBack):
    
    def after_mutation(self):
          
        print(self.alg.pop.F)

        return