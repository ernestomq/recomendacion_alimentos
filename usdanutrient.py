import os
import json
from langchain_core.documents import Document
from langchain_core.document_loaders.base import BaseLoader
from typing import List, Optional, Dict, Any

class USDANutrientLoader(BaseLoader):
    
    def __init__(
        self,
        directory_path: str,
        encoding: str = "utf-8",
        include_calculated: bool = True
    ):
        self.directory_path = directory_path
        self.encoding = encoding
        self.include_calculated = include_calculated

    def _clean_amount(self, value: Any) -> Optional[float]:
        if isinstance(value, (int,float)):
            return float(value)
        return None
    
    def _extraccion_valor_nutriente(self, nutrient_data: Dict) -> Optional[float]:
        if nutrient_data.get("amount") is not None:
            return self._clean_amount(nutrient_data["amount"])
        if nutrient_data.get("median") is not None:
            return self._clean_amount(nutrient_data["median"])
        return None
    
    MAPEO_NUTRIENTES = {
        1008: ("energy_kcal", "kcal"),
        1005: ("carbohydrate_g", "g"),
        1004: ("total_fat_g", "g"),
        1003: ("protein_g", "g"),
        1079: ("fiber_g", "g"),
        1063: ("sugars_total_g", "g"),
        1259: ("saturated_fat_g", "g"),
        1292: ("monounsaturated_fat_g", "g"),
        1293: ("polyunsaturated_fat_g", "g"),
        1258: ("trans_fat_g", "g"),
        1404: ("omega3_ala_g", "g"),
        1316: ("omega6_la_g", "g"),
        1089: ("iron_mg", "mg"),
        1090: ("magnesium_mg", "mg"),
        1091: ("phosphorus_mg", "mg"),
        1093: ("sodium_mg", "mg"),
        1092: ("potassium_mg", "mg"),
        1095: ("zinc_mg", "mg"),
        1087: ("calcium_mg", "mg"),
        1098: ("copper_mg", "mg"),
        1101: ("manganese_mg", "mg"),
        1103: ("selenium_ug", "µg"),
        1106: ("vitamin_a_rae_ug", "µg"),
        1109: ("vitamin_e_mg", "mg"),
        1162: ("vitamin_c_mg", "mg"),
        1165: ("thiamin_mg", "mg"),
        1166: ("riboflavin_mg", "mg"),
        1167: ("niacin_mg", "mg"),
        1175: ("vitamin_b6_mg", "mg"),
        1177: ("folate_ug", "µg"),
        1185: ("vitamin_k_ug", "µg"),
        1180: ("choline_mg", "mg"),
        1123: ("lutein_zeaxanthin_ug", "µg"),
        1051: ("water_g", "g"),
    }

    def _procesar_comida(self, food_data: Dict, source_filename: str) -> Optional[Document]:
        # Metadatos base
        metadata = {
            "fdc_id": food_data.get('fdcId'),
            "food_name": food_data.get('description'),
            "food_category": food_data.get('foodCategory', {}).get('description'),
            "data_type": food_data.get('dataType'),
            "publication_date": food_data.get('publicationDate'),
            "source_file": source_filename
        }

        # Si no hay nombre, este alimento no es válido
        if not metadata["food_name"]:
            return None

        nutrients_dict = {}

        # Extraer nutrientes
        for nutrient in food_data.get('foodNutrients', []):
            nutrient_id = nutrient.get('nutrient', {}).get('id')
            if nutrient_id in self.MAPEO_NUTRIENTES:
                key, unit = self.MAPEO_NUTRIENTES[nutrient_id]
                value = self._extraccion_valor_nutriente(nutrient)
                if value is not None:
                    metadata[key] = value
                    metadata[f"{key}_unit"] = unit
                    nutrients_dict[key] = value

        # Cálculos adicionales
        if self.include_calculated:
            # Carbohidratos netos
            if metadata.get('carbohydrate_g') is not None and metadata.get('fiber_g') is not None:
                net_carbs = round(metadata['carbohydrate_g'] - metadata['fiber_g'], 2)
                metadata['net_carbs_g'] = net_carbs
                nutrients_dict['net_carbs_g'] = net_carbs

            # Ratio omega-6 / omega-3
            if metadata.get('omega6_la_g') and metadata.get('omega3_ala_g'):
                if metadata['omega3_ala_g'] > 0:
                    ratio = round(metadata['omega6_la_g'] / metadata['omega3_ala_g'], 2)
                    metadata['omega6_omega3_ratio'] = ratio
                    nutrients_dict['omega6_omega3_ratio'] = ratio

        # Porciones
        portions = []
        for portion in food_data.get('foodPortions', []):
            if portion.get('gramWeight'):
                portions.append({
                    'portion_name': f"{portion.get('value', 1)} {portion.get('measureUnit', {}).get('abbreviation', '')}".strip(),
                    'gram_weight': portion.get('gramWeight')
                })

        # Guardar porción estándar (RACC) en metadata
        standard_portion = next((p for p in portions if 'RACC' in p['portion_name']), None)
        if standard_portion:
            metadata['standard_portion_g'] = standard_portion['gram_weight']
            metadata['standard_portion_name'] = standard_portion['portion_name']

        # ALMACENAR EL LLM
        content = self._format_content(
            food_name=metadata.get('food_name', 'Unknown'),
            nutrients=nutrients_dict,
            portions=portions
        )

        return Document(page_content=content, metadata=metadata)

# Una vez hemos obtenido la información de los JSON le damos un formato para el LLM
    def _format_content(self, food_name: str, nutrients: Dict, portions: List) -> str:
        lines = []
        lines.append(f"FOOD: {food_name}")
        lines.append("=" * 50)

        # Macronutrients
        lines.append(f"\n MACRONUTRIENTS (per 100g):")
        if nutrients.get('energy_kcal') is not None: 
            lines.append(f"  - Energy: {nutrients['energy_kcal']} kcal")
        if nutrients.get('protein_g') is not None: 
            lines.append(f"  - Protein: {nutrients['protein_g']}g")
        if nutrients.get('total_fat_g') is not None: 
            lines.append(f"  - Total fat: {nutrients['total_fat_g']}g")
        if nutrients.get('carbohydrate_g') is not None: 
            lines.append(f"  - Carbohydrates: {nutrients['carbohydrate_g']}g")
        if nutrients.get('fiber_g') is not None: 
            lines.append(f"  - Fiber: {nutrients['fiber_g']}g")
        if nutrients.get('net_carbs_g') is not None: 
            lines.append(f"  - Net carbs: {nutrients['net_carbs_g']}g")

        # Fat profile
        lines.append("=" * 5)
        lines.append("\n FAT PROFILE:")
        if nutrients.get('saturated_fat_g') is not None: 
            lines.append(f"  - Saturated: {nutrients['saturated_fat_g']}g")
        if nutrients.get('monounsaturated_fat_g') is not None: 
            lines.append(f"  - Monounsaturated: {nutrients['monounsaturated_fat_g']}g")
        if nutrients.get('polyunsaturated_fat_g') is not None: 
            lines.append(f"  - Polyunsaturated: {nutrients['polyunsaturated_fat_g']}g")
        if nutrients.get('omega3_ala_g') is not None: 
            lines.append(f"  - Omega-3 (ALA): {nutrients['omega3_ala_g']}g")
        if nutrients.get('omega6_la_g') is not None: 
            lines.append(f"  - Omega-6 (LA): {nutrients['omega6_la_g']}g")

        # Key minerals
        lines.append("=" * 5)
        lines.append("\n KEY MINERALS:")
        minerals_list = []
        if nutrients.get('sodium_mg') is not None: 
            minerals_list.append(f"Sodium: {nutrients['sodium_mg']}mg")
        if nutrients.get('potassium_mg') is not None: 
            minerals_list.append(f"Potassium: {nutrients['potassium_mg']}mg")
        if nutrients.get('magnesium_mg') is not None: 
            minerals_list.append(f"Magnesium: {nutrients['magnesium_mg']}mg")
        if nutrients.get('iron_mg') is not None: 
            minerals_list.append(f"Iron: {nutrients['iron_mg']}mg")
        if nutrients.get('calcium_mg') is not None: 
            minerals_list.append(f"Calcium: {nutrients['calcium_mg']}mg")
        if minerals_list:
            lines.append(f"  • {', '.join(minerals_list)}")

        # Key vitamins
        lines.append("=" * 5)
        lines.append("\n VITAMINS:")
        vitamins_list = []
        if nutrients.get('vitamin_e_mg') is not None: 
            vitamins_list.append(f"E: {nutrients['vitamin_e_mg']}mg")
        if nutrients.get('vitamin_k_ug') is not None: 
            vitamins_list.append(f"K: {nutrients['vitamin_k_ug']}µg")
        if nutrients.get('folate_ug') is not None: 
            vitamins_list.append(f"Folate: {nutrients['folate_ug']}µg")
        if nutrients.get('vitamin_b6_mg') is not None: 
            vitamins_list.append(f"B6: {nutrients['vitamin_b6_mg']}mg")
        if vitamins_list:
            lines.append(f"  - {', '.join(vitamins_list)}")

        # Standard portions
        if portions:
            lines.append("=" * 5)
            lines.append("\n STANDARD PORTIONS:")
            for p in portions[:2]:
                lines.append(f"  - {p['portion_name']}: {p['gram_weight']}g")

        # Automatic summary for the LLM (based on thresholds)
        lines.append("\n NUTRITIONAL SUMMARY:")
        summary_list = []
        if nutrients.get('fiber_g', 0) > 5:
            summary_list.append("HIGH IN FIBER")
        if nutrients.get('protein_g', 0) > 10:
            summary_list.append("GOOD SOURCE OF PROTEIN")
        if nutrients.get('sodium_mg', 0) > 400:
            summary_list.append("HIGH IN SODIUM")
        if nutrients.get('saturated_fat_g', 0) < 3 and nutrients.get('total_fat_g', 0) > 10:
            summary_list.append("PREDOMINANTLY UNSATURATED FATS")
        if summary_list:
            lines.append(f"  • {', '.join(summary_list)}")
        else:
            lines.append("  • Balanced nutritional profile")

        return "\n".join(lines)
    
    def _processar_comida_json(self, file_path: str) -> List[Document]:
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error leyendo {file_path}: {e}")
            return []
        
        filename = os.path.basename(file_path)
        docs = []

        if "FoundationFoods" in data and isinstance(data["FoundationFoods"], list):
            print(f"Procesando FoundationFoods en {filename}...")

            for food in data["FoundationFoods"]:
                doc = self._procesar_comida(food, filename)
                if doc:
                    docs.append(doc)
            return docs
        
        doc=self._procesar_comida(data, filename)
        if doc:
            docs.append(doc)
        return docs

    def load(self) -> List[Document]:
        all_docs = []
        for filename in os.listdir(self.directory_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.directory_path, filename)
                file_docs= self._processar_comida_json(file_path)
                all_docs.extend(file_docs)
                print(f"Procesados {len(file_docs)} documentos de {filename}")
        print(f"\n TOTAL: {len(all_docs)} documentos cargados")
        return all_docs