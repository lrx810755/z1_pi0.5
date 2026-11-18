"""Exercise column0 dispatch without constructing a node or sending motion."""
from pathlib import Path
from unittest.mock import patch

import horizontal50
from horizontal50_core import Settings, search_strategies


def main():
    cls=horizontal50.HorizontalAssistant
    instance=cls.__new__(cls)
    instance._collection_column=lambda:0
    with patch.object(horizontal50.base.GraspAssistant,'_generate_plan',return_value='original_plan') as generate:
        assert instance._generate_plan()=='original_plan'
        generate.assert_called_once_with()
    with patch.object(horizontal50.base.GraspAssistant,'_execute_plan',return_value='original_execute') as execute:
        assert instance._execute_plan()=='original_execute'
        execute.assert_called_once_with()
    for column in range(1,5):assert search_strategies(column)==('direct_low',)
    assert Settings().retreat_candidates_m==(.05,.07,.08,.09)
    profile=horizontal50.base.GRASP_PROFILES['beverage_bottle']
    assert profile['candidate_count']==17
    assert profile['single_alignment_tolerance_m']==(.01,.01,.01)
    print('PASS: column0 uses original planning+execution; columns1..4 use horizontal strategy; bottle calibration retained')


if __name__=='__main__':main()
