# from pydantic import BaseModel, create_model
import datetime
from enum import Enum
from typing import Dict, Optional, Union, Any
from pydantic import BaseModel as PydanticBaseModel

class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

class DataTableEnum(str, Enum):
    fixee = "fixed"
    variable = "variable"

class OptionProps(BaseModel):
    label: str
    value: str
    unit: Optional[str] = None
    provenance: Optional[str] = None

    def to_json(self):
        result = {
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "provenance": self.provenance
        }
        return result

class Field(BaseModel):
    name: str
    type: str
    timeUnit: Optional[str] = None

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Field):
            return self.name == other.name
        return False

    def to_json(self):
        result = {
            "name": self.name,
            "type": self.type,
            "timeUnit": self.timeUnit
        }
        return result

class DateRange(BaseModel):
    date_start: OptionProps
    date_end: OptionProps

    def to_json(self):
        result = {
            "date_start": self.date_start.to_json(),
            "date_end": self.date_end.to_json()
        }
        return result

class Ranges(BaseModel):
    values: list[OptionProps]
    fields: Dict[str, Union[list, DateRange]] ## Now date moved into the fields

    def to_json(self):
        result = {
            "values": [value.to_json() for value in self.values],
            "fields": {key: value.to_json() if isinstance(value, DateRange) else value for key, value in self.fields.items()}
        }
        return result

class DataPoint(BaseModel):
    tableName: str
    valueName: str ## Now valueName is the name of the field
    fields: Dict[str, Any] # Date is now moved to here

    def to_json(self):
        result = {
            "tableName": self.tableName,
            "valueName": self.valueName,
            "fields": self.fields
        }
        return result

class DataPointValue(DataPoint):
    value: float
    unit: Optional[str] = None
    
    def to_json(self):
        result = {
            "tableName": self.tableName,
            "valueName": self.valueName,
            "fields": self.fields,
            "value": self.value,
            "unit": self.unit
        }
        return result

class DataPointSet(BaseModel):
    statement: str
    tableName: str
    dataPoints: list[DataPointValue]
    fields: list[Field]
    ranges: Ranges
    reasoning: Optional[str] = None

    def to_json(self):
        result = {
            "statement": self.statement,
            "tableName": self.tableName,
            "dataPoints": [dp.to_json() for dp in self.dataPoints],
            "fields": [field.to_json() for field in self.fields],
            "ranges": self.ranges.to_json(),
            "reasoning": self.reasoning
        }
        return result

class ClaimMap(BaseModel):        
    class SuggestValue(BaseModel):
        field: str
        values: list[str]
        explain: str
        rank: int
        caution: list[str] = []

        def to_json(self) -> Any:
            return {
                "field": self.field,
                "values": self.values,
                "explain": self.explain,
                "rank": self.rank,
                "caution": self.caution,
            }

    country: list[str]
    value: list[str]
    date: list[str]
    vis: str 
    cloze_vis: str
    rephrase: str 
    suggestion: list[SuggestValue]
    mapping: Dict[str, Any]

    def to_json(self) -> Any:
        return {
            "country": self.country,
            "value": self.value,
            "date": self.date,
            "vis": self.vis,
            "cloze_vis": self.cloze_vis,
            "rephrase": self.rephrase,
            "suggestion": [sv.to_json() for sv in self.suggestion],
            "mapping": {k: list(v) if isinstance(v, set) else v for k, v in self.mapping.items()},
        }

class Dataset(BaseModel):
    name: str
    description: str
    score: float
    fields: list[str]

class UserClaimBody(BaseModel):
    userClaim: str
    paragraph: Optional[str] = None
    context: Optional[str] = 'South Korea'

class GetVizSpecBody(BaseModel):
    userClaim: str
    tableName: str
    dataPoints: list[DataPoint]

class GetVizDataBodyNew(BaseModel):
    tableName: str
    values: list[OptionProps]
    fields: Dict[str, Union[list[OptionProps], DateRange]]

class GetVizDataBodyMulti(BaseModel):
    datasets: list[Dataset]
    values: list[OptionProps]
    fields: Dict[str, Union[list[OptionProps], DateRange]]

class LogBase(BaseModel):
    event: str
    payload: Optional[str] = None
    environment: str
    client_timestamp: str
    url: str
    username: Optional[str] = None

class LogCreate(LogBase):
    pass

class Log(LogBase):
    id: int
    created_at: datetime.datetime

    class Config:
        from_attributes = True
        orm_mode = True

