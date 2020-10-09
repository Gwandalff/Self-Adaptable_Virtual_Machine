package miniJava.interpreter.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;

public interface Block extends EObject, Statement {
	EList<Statement> getStatements();

	void evaluateStatementKeepContext(State state);

	void evaluateStatement(State state);
}
