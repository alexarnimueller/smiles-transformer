import re


def split(smiles):
    pattern = r'\^|\s|#|=|-[0-9]*|\+[0-9]*|[0-9]|\[.{2,5}\]|%[0-9]{2}|\(|\)|\.|/|\\|:|@+|\{|\}|Cl|Ca|Cu|Br|Be|Ba|Bi|' \
              'Si|Se|Sr|Na|Ni|Rb|Ra|Xe|Li|Al|As|Ag|Au|Mg|Mn|Te|Zn|He|Kr|Fe|[BCFHIKNOPScnos]'
    return ' '.join(re.findall(pattern, smiles))
