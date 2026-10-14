from itg_classification import main


def test_main_prints_greeting(capsys):
    main()
    assert "itg-classification" in capsys.readouterr().out
