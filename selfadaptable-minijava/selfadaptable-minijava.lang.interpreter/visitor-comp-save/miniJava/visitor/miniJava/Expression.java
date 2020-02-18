package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Expression extends EObject, AcceptInterface, Statement, Assignee {
}
