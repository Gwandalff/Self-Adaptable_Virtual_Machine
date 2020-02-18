package miniJava.visitor.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Block extends EObject, AcceptInterface, Statement {
	EList<Statement> getStatements();
}
