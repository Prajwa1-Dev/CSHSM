import os

def map_category(source_name: str, source_category: str) -> str:
    """
    Map source categories/targets to CSHSM categories:
    Gender, Political, Religion, Geo-political, Violence, Others
    """
    if source_name == "multitarget-conan":
        cat = source_category.strip().upper()
        if cat == "WOMEN":
            return "Gender"
        elif cat == "LGBT+":
            return "Gender"
        elif cat == "JEWS":
            return "Religion"
        elif cat == "MUSLIMS":
            return "Religion"
        elif cat == "MIGRANTS":
            return "Geo-political"
        elif cat == "POC":
            return "Others"
        elif cat == "OTHER":
            return "Others"
        
    elif source_name == "conan":
        cat = source_category.strip().lower()
        if cat == "islamophobia":
            return "Religion"
            
    return "UNMAPPED"
