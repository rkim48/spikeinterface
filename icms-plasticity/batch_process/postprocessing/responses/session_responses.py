# responses/session_responses.py

from .unit_response import UnitResponse
import merge.classify_cell_type as cell_classifier


class SessionResponses:
    def __init__(self, session_path, sorting_analyzer, all_stim_timestamps, timing_params):
        self._session_path = session_path
        self._unit_responses = {}
        self._sorting_analyzer = sorting_analyzer
        self._all_stim_timestamps = all_stim_timestamps
        self._timing_params = timing_params
        self._cell_type_df = None

    @property
    def cell_type_df(self):
        if self._cell_type_df is None:
            self._cell_type_df = cell_classifier.classify_units_into_cell_types(
                self.sorting_analyzer)["df"]
        return self._cell_type_df

    def _add_unit_response(self, unit_id, unit_response):
        self._unit_responses[unit_id] = unit_response

    def get_unit_response(self, unit_id):
        return self._unit_responses.get(unit_id, None)

    def __repr__(self):
        return f"{self.__class__.__name__}(session_path={self.session_path!r}, units={len(self._unit_responses)} units)"

    # Properties for accessing private attributes
    @property
    def session_path(self):
        return self._session_path

    @property
    def unit_responses(self):
        return self._unit_responses

    @property
    def unit_ids(self):
        return list(self._unit_responses.keys())

    @property
    def sorting_analyzer(self):
        return self._sorting_analyzer

    @property
    def all_stim_timestamps(self):
        return self._all_stim_timestamps

    @property
    def timing_params(self):
        return self._timing_params
